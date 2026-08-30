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

The default 160 x 512 case is intended for interactive desktop use.  Start with
64 x 128 for quick experiments.  This is a two-dimensional laminar/unsteady
research solver, not a high-Reynolds-number engineering turbulence model.

The solver advances
`omega_t + (psi_s omega_theta - psi_theta omega_s)/H^2 = nu Delta_c omega/H^2`
and solves `Delta_c psi = -H^2 omega`, where `H=r|f'(zeta)|`.  Fourier
derivatives use multithreaded SciPy FFTs, radial derivatives use
Chebyshev--Gauss--Lobatto collocation, and time advancement uses the
three-stage LS-IMEX-RK3 coefficients in `turbo_simulator.py`.  Advection and
azimuthal diffusion are explicit; the stiff mapped radial diffusion is
implicit, so the production timestep is no longer set by the `O(Nr^-4)`
Chebyshev wall spacing.  Fourier two-thirds dealiasing is supplemented by a
smooth Chebyshev-tail filter applied only to the nonlinear product.

The default circle is centered slightly left of the origin.  Its rightmost
point is separated from the Joukowski critical point by `te_gap=0.015`, giving
a thin finite trailing edge and a strictly positive metric.  The wall is the
exact `s=0` coordinate line.  A cubic generalized Thom reconstruction supplies
wall vorticity.  Each implicit stage solves its wall coupling through a cached
dense Schur complement instead of an unstable fixed-point iteration.  Outer
vorticity is zero, with a far-boundary sponge in the last 15 percent of the
logarithmic radius.  The deterministic initial field includes a compatible
no-slip radial lift; it does not force a nominal startup ramp to completion by
timestep count.  Circulation is reported clockwise positive for the displayed
`Cl_KJ=2 Gamma/U` convention.

Pressure is reconstructed from the velocity-gradient pressure Poisson source,
with homogeneous wall-normal pressure derivative and zero outer reference.
The GUI performs cached inverse-map interpolation into a physical Cartesian
raster and uses Qt-native painting for the airfoil and Cp curves.  The bottom
chart overlays the recovered viscous surface pressure (solid) with the
analytical inviscid Kutta--Joukowski profile (dashed) for the same geometry and
angle of attack.  The status panel reports integer cumulative simulation FPS,
defined as emitted frames divided by wall time since the first Start after
Reset (pause time is included).  Reset also rejects any queued frame from the
retired worker, so a previous run cannot overwrite the new counter.

Known limitations are the finite (regularized) trailing edge, approximate wall
vorticity closure, finite outer boundary, and the remaining explicit azimuthal
and advective timestep limits.  The supplied LS recurrence is third-order for
its explicit part but only first-order for pure implicit decay; its reference
name does not imply full split order three.  Small test grids below Nr=40 use a conservative
timestep fallback because they cannot resolve the initial wall layer.  High-Re
cases require resolution; no turbulence model is supplied.  The working solver
remains the SciPy/NumPy float64 reference.  `prompt.txt` specifies correctness
and end-to-end benchmark gates for a future persistent MLX backend; FFT-only
host/device shuttling is intentionally not enabled.  Cached per-angle radial
factors scale as `O(Ntheta * Nr^2)` memory, so the largest documented grids use
several GiB even before a future device backend is considered.
