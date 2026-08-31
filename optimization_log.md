1 | relax first-pass GCROT tolerance to 3e-6 | 1.04339e-4 -> tests failed | reverted
2 | pair mirror-symmetric radial LU columns as multi-RHS solves | 1.04339e-4 -> 1.27548e-4 (+22.3%) | kept
3 | reuse one scratch slab per grouped radial LU worker | 1.27548e-4 -> 1.29967e-4 (+1.9%) | kept
4 | seed GCROT with the linear RHS instead of a full preconditioner solve | 1.29967e-4 -> 1.31553e-4 (+1.2%) | kept
5 | relax first-pass GCROT tolerance from 1e-8 to 1e-7 | 1.31553e-4 -> 1.44959e-4 (+10.2%) | kept
6 | tune first-pass GCROT tolerance from 1e-7 to 3e-7 | 1.44959e-4 -> 1.50969e-4 (+4.1%) | kept
7 | relax first-pass GCROT tolerance from 3e-7 to 1e-6 | 1.50969e-4 -> tests failed | reverted
8 | use real FFTs for real-valued theta derivatives | 1.50969e-4 -> 1.66539e-4 (+10.3%) | kept
9 | group nearby radial preconditioner profiles within 10 percent | 1.66539e-4 -> 2.12981e-4 (+27.9%) | kept
10 | assemble Thom response from cached exact near-wall inverse rows | 2.12981e-4 -> 2.58616e-4 (+21.4%) | kept
11 | balance variable-size LU groups across workers | 2.58616e-4 -> 2.67478e-4 (+3.4%) | kept
12 | tune first-pass GCROT tolerance from 3e-7 to 5e-7 | 2.67478e-4 -> 2.73378e-4 (+2.2%) | kept
13 | solve grouped LU systems in one cached permuted Fortran layout | 2.73378e-4 -> 2.85645e-4 (+4.5%) | kept
14 | fuse fixed Thom weights into the modal Poisson contraction | 2.85645e-4 -> 2.87036e-4 (+0.5%) | kept
15 | warm-start each RK stage from its last converged correction | 2.87036e-4 -> 3.01368e-4 (+5.0%) | kept
16 | extrapolate each RK-stage correction from its two-step history | 3.01368e-4 -> 3.08901e-4 (+2.5%) | kept
17 | use real FFTs for nonlinear angular de-aliasing | 3.08901e-4 -> 3.16823e-4 (+2.6%) | kept
18 | bypass zero radial-boundary forcing in Krylov preconditioning | 3.16823e-4 -> 3.35170e-4 (+5.8%) | kept
19 | use a dedicated eight-worker pool for modal Poisson LU solves | 3.35170e-4 -> 3.42349e-4 (+2.1%) | kept
20 | fuse the cached mapped-diffusion stage-matrix action | 3.42349e-4 -> 3.49759e-4 (+2.2%) | kept
21 | compile grouped-LU plans and run a fifth partition on the caller | 3.49759e-4 -> 3.56495e-4 (+1.9%) | kept
22 | transpose wall-functional FFTs for contiguous modal vector dots | 3.56495e-4 -> 3.70465e-4 (+3.9%) | kept
23 | tune first-pass GCROT tolerance from 5e-7 to 7e-7 | 3.70465e-4 -> 3.70604e-4 (+0.04%) | kept
