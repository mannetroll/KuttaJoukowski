import numpy as np

from joukowskisim.boundary import update_wall_vorticity
from joukowskisim.imex import LS_IMEX_ALPHA, LS_IMEX_BETA
from joukowskisim.solver import FlowSolver, SolverConfig


def _reference_scalar_steps(step_count, implicit_rate, explicit_rate):
    dt = 1.0 / step_count
    value = 1.0
    for _ in range(step_count):
        previous_explicit = 0.0
        for alpha, beta in zip(LS_IMEX_ALPHA, LS_IMEX_BETA):
            explicit = explicit_rate * value
            value = (
                value
                + beta * dt * implicit_rate * value
                + alpha * dt * explicit
                + beta * dt * previous_explicit
            ) / (1.0 - alpha * dt * implicit_rate)
            previous_explicit = explicit
    return value


def test_reference_ls_coefficients_and_observed_split_orders():
    assert LS_IMEX_ALPHA == (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
    assert LS_IMEX_BETA == (0.0, -17.0 / 60.0, -5.0 / 12.0)

    exact = np.exp(-1.0)
    explicit_errors = [
        abs(_reference_scalar_steps(n, 0.0, -1.0) - exact)
        for n in (40, 80)
    ]
    implicit_errors = [
        abs(_reference_scalar_steps(n, -1.0, 0.0) - exact)
        for n in (40, 80)
    ]
    # The supplied/reference recurrence is RK3 for the explicit part, but its
    # pure implicit decay is only first-order accurate.  Keep that limitation
    # measured so the LS_IMEX_RK3 name is not mistaken for full split order 3.
    assert explicit_errors[0] / explicit_errors[1] > 7.5
    assert 1.8 < implicit_errors[0] / implicit_errors[1] < 2.2


def test_stiff_radial_mode_decays_far_above_explicit_limit():
    solver = FlowSolver(SolverConfig(nr=24, ntheta=32, re=100))
    explicit_limit = solver.timestep_limits()[2]
    dt = 0.1
    assert dt > 1.0e4 * explicit_limit
    solver.radial_implicit.prepare(dt)

    x = 1.0 - 2.0 * solver.S / solver.mapping.s_max
    omega = (
        solver.S
        * (solver.mapping.s_max - solver.S)
        * np.cos(18 * np.arccos(x))
        * np.sin(2 * solver.T)
    )
    initial_norm = np.linalg.norm(omega)
    for stage, beta in enumerate(LS_IMEX_BETA):
        rhs = omega + beta * dt * solver.radial_implicit.apply(omega)
        omega = solver.radial_implicit.solve(
            rhs, stage, np.zeros(solver.config.ntheta)
        )
        assert solver.radial_implicit.residual(omega, rhs, stage) < 1e-12
        assert np.max(np.abs(omega[[0, -1]])) == 0.0

    assert np.isfinite(omega).all()
    assert np.linalg.norm(omega) < 0.98 * initial_norm


def test_cached_thom_schur_matches_direct_poisson_response():
    solver = FlowSolver(SolverConfig(nr=24, ntheta=32, re=200, alpha=7))
    dt = min(solver.stable_timestep(), 1e-4)
    solver.radial_implicit.prepare(dt)
    solver.wall_influence.prepare(dt)

    rng = np.random.default_rng(1729)
    wall = rng.standard_normal(solver.config.ntheta)
    for stage in range(3):
        response_omega = np.zeros_like(solver.omega)
        response_omega[0] = wall
        response_omega[1:-1] = (
            solver.radial_implicit.wall_response(stage) * wall[None, :]
        )
        response_psi = solver.poisson.solve(
            -solver.H2 * response_omega,
            wall=0.0,
            outer=0.0,
        )
        closed = response_omega.copy()
        update_wall_vorticity(
            closed,
            response_psi,
            solver.grid.Dss,
            solver.H2,
            solver.grid.s,
        )
        expected = wall - closed[0]
        assert np.allclose(
            solver.wall_influence.apply(stage, wall),
            expected,
            rtol=2e-11,
            atol=2e-11,
        )

    assert max(solver.wall_influence.conditions) < 1e4


def test_wall_schur_closes_stages_and_reuses_cached_factors():
    solver = FlowSolver(SolverConfig(nr=40, ntheta=64, re=200, alpha=7))
    dt = min(solver.stable_timestep(), 1e-4)
    first = solver.step(dt)
    assert first["implicit_residual"] < 1e-8
    assert first["wall_slip"] < 1e-2
    assert first["wall_iterations"] == 1
    assert first["coarse_timestep_fallback"] == 0
    assert solver.radial_implicit.factorizations == 1
    assert solver.wall_influence.assemblies == 1

    second = solver.step(dt)
    assert np.isfinite(solver.omega).all()
    assert second["implicit_residual"] < 1e-8
    assert solver.radial_implicit.factorizations == 1
    assert solver.wall_influence.assemblies == 1


def test_thom_schur_is_continuous_for_tiny_timestep():
    solver = FlowSolver(SolverConfig(nr=14, ntheta=32, re=1000, alpha=5))
    assert solver.coarse_timestep_fallback
    assert solver.stable_timestep() <= 2e-5
    initial_max = float(np.max(np.abs(solver.omega)))
    solver.step(1e-8)
    assert np.max(np.abs(solver.omega)) < 1.01 * initial_max
    assert solver.diagnostics()["implicit_residual"] < 1e-8


def test_adaptive_timestep_hysteresis_reuses_expensive_factors():
    solver = FlowSolver(SolverConfig(
        nr=40, ntheta=64, re=10000, alpha=20, cfl=0.25
    ))
    first = solver.step()
    first_dt = first["dt"]
    first_assemblies = solver.wall_influence.assemblies

    # A slightly looser safe limit should retain the accepted timestep rather
    # than rebuilding all radial and dense wall factors.
    solver.timestep_limits = lambda: (
        1.05 * first_dt,
        10.0 * first_dt,
        first["dt_diff_explicit"],
    )
    assert solver.stable_timestep() == first_dt
    solver.step()
    assert solver.wall_influence.assemblies == first_assemblies
