1 | relax first-pass GCROT tolerance to 3e-6 | 1.04339e-4 -> tests failed | reverted
2 | pair mirror-symmetric radial LU columns as multi-RHS solves | 1.04339e-4 -> 1.27548e-4 (+22.3%) | kept
3 | reuse one scratch slab per grouped radial LU worker | 1.27548e-4 -> 1.29967e-4 (+1.9%) | kept
4 | seed GCROT with the linear RHS instead of a full preconditioner solve | 1.29967e-4 -> 1.31553e-4 (+1.2%) | kept
5 | relax first-pass GCROT tolerance from 1e-8 to 1e-7 | 1.31553e-4 -> 1.44959e-4 (+10.2%) | kept
6 | tune first-pass GCROT tolerance from 1e-7 to 3e-7 | 1.44959e-4 -> 1.50969e-4 (+4.1%) | kept
