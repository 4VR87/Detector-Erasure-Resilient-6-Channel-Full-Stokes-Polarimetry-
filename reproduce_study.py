import itertools, json, math, platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; FIG=ROOT/'figures'
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED=20260919
rng=np.random.default_rng(SEED)

PROPOSED=np.array([
[-0.81571441,-0.52203330, 0.24918113],
[ 0.06262398, 0.15828331, 0.98540582],
[ 0.55801114, 0.57312615,-0.60012497],
[-0.12094172,-0.60542375,-0.78666078],
[ 0.73361965,-0.64213493, 0.22240715],
[-0.52176286, 0.84765388,-0.09615830],
],float)
OCTA=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
CUBE=np.array(list(itertools.product([-1,1],repeat=3)),float)/np.sqrt(3)
TETRA=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)/np.sqrt(3)
DESIGNS={'Tetra-4':TETRA,'Octa-6':OCTA,'Proposed-6':PROPOSED,'Cube-8':CUBE}


def Wmat(a):
    return 0.5*np.column_stack([np.ones(len(a)),a])

def stokes_projection(x):
    t0=float(x[0]); v=np.asarray(x[1:],float); nv=np.linalg.norm(v)
    if t0>=nv and t0>=0: return x.copy()
    if nv<=-t0: return np.zeros(4)
    if nv==0: return np.array([max(t0,0),0,0,0],float)
    alpha=0.5*(t0+nv)
    return np.r_[alpha,alpha*v/nv]

def random_states(n,rng):
    dirs=rng.normal(size=(n,3)); dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    dop=rng.random(n)
    s=dirs*dop[:,None]
    return np.c_[np.ones(n),s],dop

def estimate_one(Wk,zk,yk,scale,sigma_read,project=True):
    if np.linalg.matrix_rank(Wk)<4:
        return None,False
    var=(np.maximum(yk,1.0)+sigma_read**2)/(scale**2)
    wt=1.0/var
    M=Wk.T@(wt[:,None]*Wk)
    b=Wk.T@(wt*zk)
    try: x=np.linalg.solve(M,b)
    except np.linalg.LinAlgError: x=np.linalg.lstsq(Wk,zk,rcond=None)[0]
    invalid=(x[0]<=0) or (np.linalg.norm(x[1:])>x[0])
    if project: x=stokes_projection(x)
    return x,invalid

def project_batch(X):
    X=np.asarray(X,float).copy(); t0=X[:,0]; V=X[:,1:]; nv=np.linalg.norm(V,axis=1)
    inside=(t0>=nv)&(t0>=0); zero=(nv<=-t0); mid=~inside&~zero
    X[zero]=0.0
    if np.any(mid):
        alpha=0.5*(t0[mid]+nv[mid]); X[mid,0]=alpha; X[mid,1:]=alpha[:,None]*V[mid]/nv[mid,None]
    return X

def simulate(a,n_states=12000,photons=60000,sigma_read=3.0,dropout=0,seed=1,project=True,actual_a=None):
    rg=np.random.default_rng(seed)
    S,dop=random_states(n_states,rg)
    n=len(a); Wnom=Wmat(a); Wact=Wmat(a if actual_a is None else actual_a)
    scale=photons/n
    lam=scale*(S@Wact.T)
    y=rg.poisson(np.clip(lam,0,None)).astype(float)+rg.normal(0,sigma_read,size=lam.shape)
    z=y/scale
    keeps=list(itertools.combinations(range(n),n-dropout))
    which=rg.integers(len(keeps),size=n_states)
    success=np.ones(n_states,dtype=bool); invalid=np.zeros(n_states,dtype=bool); Xhat=np.zeros((n_states,4),float)
    for ci,kt in enumerate(keeps):
        idx=np.where(which==ci)[0]
        if idx.size==0: continue
        keep=np.asarray(kt); Ws=Wnom[keep]
        if np.linalg.matrix_rank(Ws)<4:
            success[idx]=False; continue
        yi=y[np.ix_(idx,keep)]; zi=z[np.ix_(idx,keep)]
        var=(np.maximum(yi,1.0)+sigma_read**2)/(scale**2); wt=1.0/var
        M=np.einsum('ki,nk,kj->nij',Ws,wt,Ws)
        b=np.einsum('ki,nk->ni',Ws,wt*zi)
        try: x=np.linalg.solve(M,b[...,None])[...,0]
        except np.linalg.LinAlgError:
            pin=np.linalg.pinv(Ws); x=zi@pin.T
        invalid[idx]=(x[:,0]<=0)|(np.linalg.norm(x[:,1:],axis=1)>x[:,0])
        if project: x=project_batch(x)
        bad=x[:,0]<=1e-12; success[idx[bad]]=False
        Xhat[idx]=x
    ok=np.where(success)[0]
    sh=Xhat[ok,1:]/Xhat[ok,0,None]; st=S[ok,1:]; dp=dop[ok]
    ev=np.linalg.norm(sh-st,axis=1); doper=np.abs(np.linalg.norm(sh,axis=1)-dp)
    mask=(dp>=0.2)&(np.linalg.norm(sh,axis=1)>1e-12)
    if np.any(mask):
        ca=np.sum(sh[mask]*st[mask],axis=1)/(np.linalg.norm(sh[mask],axis=1)*dp[mask]); ca=np.clip(ca,-1,1); ang=np.degrees(np.arccos(ca))
    else: ang=np.array([])
    return {
        'rmse_quv': float(np.sqrt(np.mean(ev**2))) if ev.size else np.nan,
        'median_quv_error': float(np.median(ev)) if ev.size else np.nan,
        'p95_quv_error': float(np.percentile(ev,95)) if ev.size else np.nan,
        'dop_mae': float(np.mean(doper)) if doper.size else np.nan,
        'median_poincare_angle_deg': float(np.median(ang)) if ang.size else np.nan,
        'availability': float(ok.size/n_states),
        'preprojection_unphysical_fraction': float(invalid[ok].mean()) if ok.size else np.nan,
        'n_success': int(ok.size)
    }

def perturb_vectors(a,sigma_deg,rng):
    if sigma_deg==0: return a.copy()
    out=[]; sig=np.deg2rad(sigma_deg)
    for v in a:
        d=rng.normal(size=3); d-=np.dot(d,v)*v
        nd=np.linalg.norm(d)
        if nd<1e-12: d=np.array([v[1],-v[0],0.0]); nd=np.linalg.norm(d)
        d/=nd
        ang=rng.normal(0,sig)
        out.append(v*np.cos(ang)+d*np.sin(ang))
    return np.asarray(out)

# analyzer-state table
rows=[]
for name,a in DESIGNS.items():
    for i,v in enumerate(a,1):
        psi=0.5*np.degrees(np.arctan2(v[1],v[0]))
        chi=0.5*np.degrees(np.arcsin(np.clip(v[2],-1,1)))
        rows.append([name,i,*v,psi,chi])
pd.DataFrame(rows,columns=['design','channel','a1','a2','a3','azimuth_psi_deg','ellipticity_chi_deg']).to_csv(DATA/'analyzer_states.csv',index=False)

# subset condition/rank audit
rows=[]
for name,a in DESIGNS.items():
    A=np.column_stack([np.ones(len(a)),a])
    for drop in range(0,min(2,len(a)-4)+1):
        for dropped in itertools.combinations(range(len(a)),drop):
            keep=[i for i in range(len(a)) if i not in dropped]
            M=A[keep]
            r=np.linalg.matrix_rank(M)
            c=np.linalg.cond(M) if r==4 else np.inf
            rows.append([name,drop,';'.join(str(i+1) for i in dropped),r,c])
cond_df=pd.DataFrame(rows,columns=['design','n_dropouts','dropped_channels','rank','condition_number'])
cond_df.to_csv(DATA/'subset_condition_numbers.csv',index=False)

# Nominal main comparison
rows=[]
for name,a in DESIGNS.items():
    for d in [0,1,2]:
        if len(a)-d<4: continue
        r=simulate(a,n_states=30000,photons=60000,sigma_read=3,dropout=d,seed=SEED+31*d+len(a),project=True)
        rows.append({'design':name,'dropouts':d,'photons_total':60000,'sigma_read_e':3.0,**r})
main=pd.DataFrame(rows); main.to_csv(DATA/'main_comparison.csv',index=False)

# projection ablation for proposed under two dropouts
rows=[]
for pr in [False,True]:
    for seed in range(5):
        r=simulate(PROPOSED,n_states=10000,photons=60000,sigma_read=3,dropout=2,seed=SEED+500+seed,project=pr)
        rows.append({'projection':pr,'trial':seed+1,**r})
ab=pd.DataFrame(rows); ab.to_csv(DATA/'projection_ablation.csv',index=False)

# Photon sweep
rows=[]
for photons in [3000,10000,30000,60000,100000]:
    for name in ['Octa-6','Proposed-6','Cube-8']:
        a=DESIGNS[name]
        r=simulate(a,n_states=12000,photons=photons,sigma_read=3,dropout=2,seed=SEED+photons//100+len(a),project=True)
        rows.append({'design':name,'photons_total':photons,'dropouts':2,**r})
phot=pd.DataFrame(rows); phot.to_csv(DATA/'photon_sweep.csv',index=False)

# Misalignment sensitivity: static analyzer perturbation per trial, reconstruction nominal
rows=[]
for sig in [0,0.25,0.5,1.0,2.0]:
    for design_name in ['Proposed-6','Cube-8']:
        a=DESIGNS[design_name]
        for tr in range(10):
            rg=np.random.default_rng(SEED+10000+tr+int(sig*100)*31+len(a))
            aa=perturb_vectors(a,sig,rg)
            r=simulate(a,n_states=4000,photons=60000,sigma_read=3,dropout=2,seed=SEED+20000+tr+int(sig*100)*17+len(a),project=True,actual_a=aa)
            rows.append({'design':design_name,'sigma_analyzer_deg':sig,'trial':tr+1,**r})
mis=pd.DataFrame(rows); mis.to_csv(DATA/'misalignment_trials.csv',index=False)
mis_summary=mis.groupby(['design','sigma_analyzer_deg']).agg(rmse_mean=('rmse_quv','mean'),rmse_sd=('rmse_quv','std'),availability_mean=('availability','mean'),angle_mean_deg=('median_poincare_angle_deg','mean')).reset_index()
mis_summary.to_csv(DATA/'misalignment_sensitivity.csv',index=False)

# Repeated nominal two-dropout trials for CIs
rows=[]
for name in ['Octa-6','Proposed-6','Cube-8']:
    a=DESIGNS[name]
    for tr in range(10):
        r=simulate(a,n_states=10000,photons=60000,sigma_read=3,dropout=2,seed=SEED+30000+100*len(a)+tr,project=True)
        rows.append({'design':name,'trial':tr+1,**r})
rep=pd.DataFrame(rows); rep.to_csv(DATA/'repeated_trials.csv',index=False)
summary=[]
for name,g in rep.groupby('design'):
    n=len(g); crit=t.ppf(0.975,n-1)
    for metric in ['rmse_quv','dop_mae','median_poincare_angle_deg','availability']:
        mean=g[metric].mean(); sd=g[metric].std(ddof=1); half=crit*sd/math.sqrt(n)
        summary.append([name,metric,mean,sd,mean-half,mean+half,n])
pd.DataFrame(summary,columns=['design','metric','mean','sd','ci95_low','ci95_high','n_trials']).to_csv(DATA/'repeated_trial_summary.csv',index=False)

# noiseless exactness check and worst subset reconstruction residual
rows=[]
for name,a in DESIGNS.items():
    W=Wmat(a)
    S,_=random_states(1000,np.random.default_rng(SEED+77+len(a)))
    for d in [0,1,2]:
        if len(a)-d<4: continue
        max_res=0; bad=0
        for dropped in itertools.combinations(range(len(a)),d):
            keep=[i for i in range(len(a)) if i not in dropped]
            if np.linalg.matrix_rank(W[keep])<4:
                bad+=1; continue
            Z=S@W[keep].T
            Sh=Z@np.linalg.pinv(W[keep]).T
            max_res=max(max_res,float(np.max(np.abs(Sh-S))))
        rows.append([name,d,bad,max_res])
pd.DataFrame(rows,columns=['design','dropouts','rank_deficient_patterns','max_abs_noiseless_reconstruction_error']).to_csv(DATA/'independent_validation.csv',index=False)

# Representative state-wise subset noise test for Proposed-6 (every 2-dropout pair)
rg=np.random.default_rng(SEED+909)
S,_=random_states(6000,rg)
W=Wmat(PROPOSED); scale=60000/6
lam=scale*(S@W.T); y=rg.poisson(lam).astype(float)+rg.normal(0,3,size=lam.shape); z=y/scale
rows=[]
for dropped in itertools.combinations(range(6),2):
    keep=np.array([i for i in range(6) if i not in dropped])
    errs=[]
    for j in range(len(S)):
        x,_=estimate_one(W[keep],z[j,keep],y[j,keep],scale,3,True)
        sh=x[1:]/x[0]; errs.append(np.linalg.norm(sh-S[j,1:]))
    rows.append([';'.join(str(i+1) for i in dropped),np.linalg.cond(np.column_stack([np.ones(6),PROPOSED])[keep]),np.sqrt(np.mean(np.square(errs))),np.percentile(errs,95)])
pd.DataFrame(rows,columns=['dropped_channels','condition_number','rmse_quv','p95_quv_error']).to_csv(DATA/'proposed_subset_noise.csv',index=False)

# ----- Figures -----
plt.rcParams.update({'font.size':9,'figure.dpi':180})
# Fig1 schematic
fig,ax=plt.subplots(figsize=(8,3.2)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
ax.annotate('',xy=(2.0,3),xytext=(0.6,3),arrowprops=dict(arrowstyle='->',lw=1.5)); ax.text(0.55,3.35,'Input Stokes beam',ha='left')
ax.add_patch(plt.Rectangle((2.0,2.25),1.3,1.5,fill=False,lw=1.2)); ax.text(2.65,3,'1→6\nbeam splitter',ha='center',va='center')
ys=np.linspace(0.65,5.35,6)
for i,y0 in enumerate(ys):
    ax.plot([3.3,4.4],[3,y0],lw=0.9)
    ax.add_patch(plt.Rectangle((4.4,y0-0.23),1.35,0.46,fill=False,lw=0.9)); ax.text(5.075,y0,f'Analyzer {i+1}',ha='center',va='center',fontsize=8)
    ax.plot([5.75,6.45],[y0,y0],lw=0.9); ax.plot(6.55,y0,'s',ms=5)
ax.text(6.55,5.75,'detector channels',ha='center',fontsize=8)
ax.annotate('',xy=(8.0,3),xytext=(6.8,3),arrowprops=dict(arrowstyle='->',lw=1.2)); ax.add_patch(plt.Rectangle((8.0,2.15),1.55,1.7,fill=False,lw=1.2)); ax.text(8.78,3,'erasure-aware\nWLS + physical\nStokes projection',ha='center',va='center',fontsize=8)
ax.text(4.9,0.05,'Analyzer states are optimized so every four-channel survivor subset remains full rank.',ha='center',fontsize=8)
fig.tight_layout(); fig.savefig(FIG/'Fig1_system_schematic.png',bbox_inches='tight'); fig.savefig(FIG/'Fig1_system_schematic.pdf',bbox_inches='tight'); plt.close(fig)

# Fig2 Poincare points
fig=plt.figure(figsize=(7.2,3.4))
ax1=fig.add_subplot(121,projection='3d'); ax2=fig.add_subplot(122,projection='3d')
for ax,a,title in [(ax1,OCTA,'(a) Octahedral 6-state baseline'),(ax2,PROPOSED,'(b) Proposed erasure-robust 6-state')]:
    u=np.linspace(0,2*np.pi,40); v=np.linspace(0,np.pi,20); x=np.outer(np.cos(u),np.sin(v)); yy=np.outer(np.sin(u),np.sin(v)); zz=np.outer(np.ones_like(u),np.cos(v)); ax.plot_wireframe(x,yy,zz,rstride=5,cstride=5,linewidth=0.25,alpha=0.25)
    ax.scatter(a[:,0],a[:,1],a[:,2],s=30)
    for i,p in enumerate(a): ax.text(*(p*1.09),str(i+1),fontsize=8)
    ax.set(xlabel='$a_1$',ylabel='$a_2$',zlabel='$a_3$',xlim=(-1,1),ylim=(-1,1),zlim=(-1,1)); ax.set_title(title,fontsize=9)
fig.tight_layout(); fig.savefig(FIG/'Fig2_poincare_states.png',bbox_inches='tight'); fig.savefig(FIG/'Fig2_poincare_states.pdf',bbox_inches='tight'); plt.close(fig)

# Fig3 subset conditioning
def two_drop_conds(a):
    A=np.column_stack([np.ones(len(a)),a]); vals=[]
    for drop in itertools.combinations(range(len(a)),2):
        keep=[i for i in range(len(a)) if i not in drop]; r=np.linalg.matrix_rank(A[keep]); vals.append(np.linalg.cond(A[keep]) if r==4 else np.nan)
    return np.array(vals)
co=two_drop_conds(OCTA); cp=two_drop_conds(PROPOSED); cc=two_drop_conds(CUBE)
fig,ax=plt.subplots(figsize=(6.5,3.6));
ax.scatter(np.arange(1,len(cp)+1),np.sort(cp),label='Proposed-6',marker='o')
finite=np.sort(co[np.isfinite(co)]); ax.scatter(np.arange(1,len(finite)+1),finite,label='Octa-6 finite subsets',marker='s')
ax.scatter(np.arange(1,len(cc)+1),np.sort(cc),label='Cube-8',marker='^',s=18)
ax.axhline(10,ls='--',lw=0.8); ax.text(1,10.25,'condition number 10',fontsize=8)
ax.text(9.5,8.8,'Octa-6: 3 of 15\ntwo-erasure patterns are rank deficient',fontsize=8)
ax.set_xlabel('Subset index after sorting'); ax.set_ylabel('2-norm condition number'); ax.set_ylim(1,11); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/'Fig3_subset_conditioning.png',bbox_inches='tight'); fig.savefig(FIG/'Fig3_subset_conditioning.pdf',bbox_inches='tight'); plt.close(fig)

# Fig4 photon sweep
fig,ax=plt.subplots(figsize=(6.4,3.6))
for name,marker in [('Octa-6','s'),('Proposed-6','o'),('Cube-8','^')]:
    g=phot[phot.design==name]; ax.plot(g.photons_total,g.rmse_quv,marker=marker,label=name)
ax.set_xscale('log'); ax.set_xlabel('Total incident photon budget'); ax.set_ylabel('RMSE of normalized $(q,u,v)$'); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/'Fig4_photon_sweep.png',bbox_inches='tight'); fig.savefig(FIG/'Fig4_photon_sweep.pdf',bbox_inches='tight'); plt.close(fig)

# Fig5 misalignment
fig,ax=plt.subplots(figsize=(6.4,3.6))
for name,marker in [('Proposed-6','o'),('Cube-8','^')]:
    g=mis_summary[mis_summary.design==name]; ax.errorbar(g.sigma_analyzer_deg,g.rmse_mean,yerr=g.rmse_sd,marker=marker,capsize=3,label=name)
ax.set_xlabel('RMS analyzer-state angular perturbation (deg)'); ax.set_ylabel('RMSE of normalized $(q,u,v)$'); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/'Fig5_misalignment_sensitivity.png',bbox_inches='tight'); fig.savefig(FIG/'Fig5_misalignment_sensitivity.pdf',bbox_inches='tight'); plt.close(fig)

# Fig6 main comparison bars
m2=main[main.dropouts==2].copy()
fig,ax=plt.subplots(figsize=(6.5,3.6)); x=np.arange(len(m2)); w=.38
ax.bar(x-w/2,m2.rmse_quv,w,label='RMSE'); ax2=ax.twinx(); ax2.bar(x+w/2,m2.availability,w,label='availability',alpha=.45)
ax.set_xticks(x,m2.design,rotation=15); ax.set_ylabel('RMSE of normalized $(q,u,v)$'); ax2.set_ylabel('Reconstruction availability'); ax2.set_ylim(0,1.08)
lines,labels=ax.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels(); ax.legend(lines+lines2,labels+labels2,fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(FIG/'Fig6_two_dropout_tradeoff.png',bbox_inches='tight'); fig.savefig(FIG/'Fig6_two_dropout_tradeoff.pdf',bbox_inches='tight'); plt.close(fig)

# software info
(ROOT/'reproducibility'/'software_versions.txt').write_text(f'Python {platform.python_version()}\nNumPy {np.__version__}\npandas {pd.__version__}\nSciPy imported by optimization/reconstruction\nMatplotlib {plt.matplotlib.__version__}\n',encoding='utf-8')
(ROOT/'reproducibility'/'config.json').write_text(json.dumps({'random_seed':SEED,'main_states':30000,'repeated_trials':10,'states_per_repeated_trial':10000,'nominal_total_photons':60000,'read_noise_e_rms':3.0,'dropouts_of_interest':2,'proposed_analyzers':PROPOSED.tolist()},indent=2),encoding='utf-8')
print('Study reproduced in',ROOT)
