# Detector-Erasure-Resilient-6-Channel-Full-Stokes-Polarimetry-
A fault-tolerant six-channel full-Stokes polarimeter is developed using worst-subset Poincaré-state optimization. Differential evolution minimizes the maximum condition number under any two-channel erasures, preserving full-rank reconstruction while balancing noise efficiency against detector count.

## Measurable contributions
1. A worst-survivor-subset Poincare-state objective that minimizes the maximum condition number over all four-channel survivor subsets of a six-channel analyzer, with a secondary complete-matrix conditioning term.
2. A reproducible six-state numerical analyzer solution for which all 15 possible two-erasure survivor subsets are full rank; the conventional octahedral six-state baseline contains three rank-deficient two-erasure subsets.
3. An erasure-aware mixed Poisson-Gaussian weighted least-squares reconstruction followed by projection onto the physically admissible Stokes cone.
4. Quantitative comparison with four-state tetrahedral, six-state octahedral, and eight-state cubic analyzers under photon noise, read noise, channel erasure, photon-budget variation, and analyzer-state perturbation.
5. Independent noiseless rank/reconstruction checks, repeated Monte Carlo trials, raw CSV outputs, executable Python code, and figure-generation scripts.
