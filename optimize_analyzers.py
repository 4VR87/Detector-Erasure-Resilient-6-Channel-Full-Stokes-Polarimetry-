import itertools, json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
ROOT=Path(__file__).resolve().parents[1]

def matrix_from_angles(x):
    n=len(x)//2; th=np.asarray(x[:n]); ph=np.asarray(x[n:])
    a=np.column_stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    return np.column_stack([np.ones(n),a]),a

def objective(x):
    A,_=matrix_from_angles(x); vals=[]
    for idx in itertools.combinations(range(6),4):
        M=A[list(idx)]
        if np.linalg.matrix_rank(M)<4: return 1e6
        vals.append(np.linalg.cond(M,2))
    return max(vals)+0.01*np.linalg.cond(A,2)

if __name__=='__main__':
    bounds=[(0,np.pi)]*6+[(-np.pi,np.pi)]*6
    res=differential_evolution(objective,bounds,seed=17,popsize=10,maxiter=300,tol=1e-6,polish=True,workers=1,updating='immediate')
    A,a=matrix_from_angles(res.x)
    vals=[np.linalg.cond(A[list(idx)],2) for idx in itertools.combinations(range(6),4)]
    out={'objective':float(res.fun),'worst_two_erasure_condition':float(max(vals)),'full_condition':float(np.linalg.cond(A)),'analyzer_vectors':a.tolist(),'optimizer_seed':17,'maxiter':300,'popsize':10}
    (ROOT/'data'/'optimization_result.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
