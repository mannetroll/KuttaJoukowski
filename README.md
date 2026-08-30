# JoukowskiSim

JoukowskiSim solves the two-dimensional incompressible vorticity equation on a
body-fitted conformal grid.  The auxiliary domain is the exterior of an offset
circle, with logarithmic radius and a periodic angle.  A regularized Joukowski
map turns the inner coordinate line into a finite-thickness airfoil.

```bash
uv sync --extra test
uv run simulation
uv run python -m joukowskisim.main --headless --steps 100
uv run pytest -q
```

The default Re=10000, 240 x 768 case is intended for resolved production runs.  Start with
64 x 128 for quick experiments.  This is a two-dimensional laminar/unsteady
research solver, not a high-Reynolds-number engineering turbulence model.

The solver advances
`omega_t + (psi_s omega_theta - psi_theta omega_s)/H^2 = nu Delta_c omega/H^2`
and solves `Delta_c psi = -H^2 omega`, where `H=r|f'(zeta)|`.  Fourier
derivatives use multithreaded SciPy FFTs, radial derivatives use
Chebyshev--Gauss--Lobatto collocation, and time advancement uses the
three-stage LS-IMEX-RK3 coefficients in `turbo_simulator.py`.  Advection and
far-field sponge damping are explicit; the complete variable-metric viscous
operator `nu H^-2 (d_ss + d_thetatheta)` is implicit.  Each stage uses a
matrix-free GCROT solve of the coupled two-dimensional operator.  Its cached
radial/Thom preconditioner shares representative LU factors only between
nearby theta columns whose complete radial coefficient profiles differ by at
most 10 percent.  This approximation affects only the preconditioner; the
coupled operator and residual remain exact.  A cached two-row Poisson
functional applies the wall equation cheaply inside Krylov products.
Consequently the production timestep is no longer set by either the
`O(Nr^-4)` Chebyshev wall spacing or azimuthal diffusion.  Fourier two-thirds
dealiasing is supplemented by a smooth Chebyshev-tail filter applied only to
the nonlinear product.
Independent cached Poisson and radial LU applications are dispatched as four
coarse LAPACK worker batches instead of thousands of serial Python calls.  Set
`JOUKOWSKISIM_LU_WORKERS` to override that default for benchmarking.

The default circle is centered slightly left of the origin.  Its rightmost
point is separated from the Joukowski critical point by `te_gap=0.015`, giving
a thin finite trailing edge and a strictly positive metric.  The wall is the
exact `s=0` coordinate line.  A cubic generalized Thom reconstruction supplies
wall vorticity.  Each full implicit stage includes the Thom relation as an
algebraic row; its cached dense radial Schur complement preconditions that
coupled solve instead of using an unstable wall fixed-point iteration.  Outer
vorticity is zero, with a far-boundary sponge in the last 15 percent of the
logarithmic radius.  The deterministic initial field includes a compatible
no-slip radial lift; it does not force a nominal startup ramp to completion by
timestep count.  Circulation is reported clockwise positive on a documented
near-wall `s=constant` contour, with `Cl(Gamma)=2 Gamma/U`.  In viscous
unsteady flow this contour diagnostic is not automatically the inviscid Kutta
circulation.

Pressure is reconstructed from the velocity-gradient pressure Poisson source,
with homogeneous wall-normal pressure derivative and zero outer reference.
The GUI performs cached inverse-map interpolation into a physical Cartesian
raster and uses Qt-native painting for the airfoil and Cp curves.  The bottom
chart overlays the recovered viscous surface pressure (solid) with the
analytical inviscid Kutta--Joukowski profile (dashed) for the same geometry and
angle of attack.  The status panel reports integer cumulative simulation FPS,
defined as emitted frames divided by wall time since the first Start after
Reset (pause time is included), plus simulation-time units advanced per wall
second.  It also shows the current local DateTime and monotonic wall time since
Start; both continue updating while paused, and Reset returns wall time to
`00:00:00`.  Reset also rejects any queued frame from the retired worker, so a
previous run cannot overwrite the new counter.
The interactive default emits every timestep; pressure remains recovered every
20 steps.  Physical-raster interpolation uses cached flat indices and bilinear
weights.  On the development M1 Max, the full implicit solve lowers a 30-frame
160 x 512 GUI-equivalent reference loop from roughly 15 raw FPS to 2.40 FPS, but
each step initially advances 27.8 times more physical time.  The measured
simulation-time throughput is about 6.6 times higher, which is the relevant
time-to-solution metric.  A stable 100-step headless run advanced to
`t=0.01491` in 35.88 seconds (`4.16e-4` simulation-time units per wall second).

Known limitations are the finite (regularized) trailing edge, approximate wall
vorticity closure, finite outer boundary, and the remaining advective timestep
limit.  The coupled iterative stage is costlier per step and uses a measured
residual tolerance.  Krylov vectors are deliberately not retained across
timesteps because measured stale-subspace overhead exceeded reuse gains.  The
supplied LS recurrence is third-order for its explicit part but
only first-order for pure implicit decay; its reference name does not imply
full split order three.  Small test grids below Nr=40 use a conservative
timestep fallback because they cannot resolve the initial wall layer.  High-Re
cases require resolution; no turbulence model is supplied.  The working solver
remains the SciPy/NumPy float64 reference.  `prompt.txt` specifies correctness
and end-to-end benchmark gates for a future persistent MLX backend; FFT-only
host/device shuttling is intentionally not enabled.  Cached per-angle radial
factors scale as `O(Ntheta * Nr^2)` memory, so the largest documented grids use
several GiB even before a future device backend is considered.

`FlowSolver.omega` and `FlowSolver.psi` form one coupled internal state.  Do
not mutate either array in place: doing so bypasses Poisson/Thom compatibility
and velocity-cache invalidation.  Create a new solver for a new initial state
unless a synchronization API is added for that experiment.
