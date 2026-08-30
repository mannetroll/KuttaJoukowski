import numpy as np

from joukowskisim.boundary import (
    update_wall_vorticity,
    wall_vorticity_from_streamfunction,
)
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


def test_radial_factors_pair_only_tightly_symmetric_metric_columns():
    symmetric = FlowSolver(SolverConfig(nr=16, ntheta=32, camber=0.0))
    expected_pairs = (
        ((0,),)
        + tuple((column, 32 - column) for column in range(1, 16))
        + ((16,),)
    )
    assert symmetric.radial_implicit._factor_groups == expected_pairs

    cambered = FlowSolver(SolverConfig(nr=16, ntheta=32, camber=0.01))
    assert cambered.radial_implicit._factor_groups == tuple(
        (column,) for column in range(32)
    )

    dt = 1e-5
    symmetric.radial_implicit.prepare(dt)
    assert all(
        len(factors) == len(expected_pairs)
        for factors in symmetric.radial_implicit._factors
    )

    rng = np.random.default_rng(2718)
    rhs = rng.standard_normal(symmetric.omega.shape)
    wall = rng.standard_normal(symmetric.config.ntheta)
    outer = rng.standard_normal(symmetric.config.ntheta)
    original_rhs = rhs.copy()
    result = symmetric.radial_implicit.solve(rhs, 1, wall, outer)
    assert symmetric.radial_implicit.residual(result, rhs, 1) < 2e-12
    assert np.array_equal(rhs, original_rhs)


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
    # The complete variable-metric stage is now an iterative 2-D solve.  Its
    # direct collocation residual includes cancellation in the largest
    # Chebyshev rows, so use a tolerance commensurate with the Krylov solve.
    assert first["implicit_residual"] < 1e-6
    assert first["wall_slip"] < 1e-2
    assert first["wall_iterations"] == 1
    assert first["coarse_timestep_fallback"] == 0
    assert solver.radial_implicit.factorizations == 1
    assert solver.wall_influence.assemblies == 1

    second = solver.step(dt)
    assert np.isfinite(solver.omega).all()
    assert second["implicit_residual"] < 1e-6
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


def test_full_implicit_apply_matches_mapped_coordinate_laplacian():
    solver = FlowSolver(SolverConfig(nr=16, ntheta=32, re=80))
    field = (
        solver.S
        * (solver.mapping.s_max - solver.S)
        * np.sin(5 * solver.T)
    )
    expected = (
        solver.config.nu
        * (
            solver.grid.Dss @ field
            + solver.grid.theta_derivative(field, 2)
        )
        / solver.H2
    )
    expected[[0, -1]] = 0.0
    assert np.allclose(
        solver.implicit_diffusion.apply(field),
        expected,
        rtol=2e-13,
        atol=2e-13,
    )

    solver.sponge_strength[:] = 0.0
    solver.nonlinear_term = lambda omega, psi: np.zeros_like(omega)
    assert np.array_equal(
        solver._explicit_rhs(field, np.zeros_like(field)),
        np.zeros_like(field),
    )


def test_fast_wall_poisson_functional_matches_complete_response():
    solver = FlowSolver(SolverConfig(nr=18, ntheta=32, re=100))
    rng = np.random.default_rng(419)
    field = rng.standard_normal(solver.omega.shape)
    response = solver.poisson.solve(
        -solver.H2 * field,
        wall=0.0,
        outer=0.0,
    )
    expected = wall_vorticity_from_streamfunction(
        response,
        solver.H2,
        solver.grid.s,
    )
    assert np.allclose(
        solver._wall_response_from_vorticity(field),
        expected,
        rtol=3e-11,
        atol=3e-11,
    )


def test_full_stage_matches_direct_dense_solve_on_tiny_grid():
    solver = FlowSolver(SolverConfig(nr=8, ntheta=8, re=50, alpha=0))
    dt = 2e-4
    stage = 1
    solver.radial_implicit.prepare(dt)
    solver.wall_influence.prepare(dt)
    alpha_dt = LS_IMEX_ALPHA[stage] * dt
    size = solver.omega.size
    matrix = np.empty((size, size))
    for column in range(size):
        basis = np.zeros(size)
        basis[column] = 1.0
        matrix[:, column] = solver._implicit_operator_apply(
            basis.reshape(solver.omega.shape),
            alpha_dt,
        ).ravel()

    rng = np.random.default_rng(8675309)
    rhs = rng.standard_normal(solver.omega.shape)
    rhs[[0, -1]] = 0.0
    expected = np.linalg.solve(matrix, rhs.ravel()).reshape(rhs.shape)
    zero = np.zeros_like(rhs)
    got, psi, residual, _, _ = solver._implicit_stage(
        rhs,
        stage,
        zero,
        zero,
    )
    assert np.allclose(got, expected, rtol=2e-8, atol=2e-9)
    assert residual < 2e-8
    closed = got.copy()
    update_wall_vorticity(
        closed,
        psi,
        solver.grid.Dss,
        solver.H2,
        solver.grid.s,
    )
    assert np.max(np.abs(closed[0] - got[0])) < 2e-8


def test_stiff_angular_mode_decays_above_old_explicit_limit():
    solver = FlowSolver(SolverConfig(nr=40, ntheta=64, re=10, alpha=0))
    old_theta_limit = solver.timestep_limits()[1]
    dt = 100.0 * old_theta_limit
    solver.radial_implicit.prepare(dt)
    solver.wall_influence.prepare(dt)
    x = 1.0 - 2.0 * solver.S / solver.mapping.s_max
    rhs = (
        solver.S
        * (solver.mapping.s_max - solver.S)
        * (1.0 + 0.1 * np.cos(3 * np.arccos(x)))
        * np.sin(20 * solver.T)
    )
    zero = np.zeros_like(rhs)
    result, _, residual, _, _ = solver._implicit_stage(
        rhs,
        2,
        zero,
        zero,
    )
    assert dt >= 100.0 * old_theta_limit
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) < np.linalg.norm(rhs)
    assert residual < 1e-6


def test_resolved_timestep_ignores_counterfactual_diffusion_limits():
    solver = FlowSolver(SolverConfig(nr=40, ntheta=64, re=1000))
    solver.timestep_limits = lambda: (2e-3, 1e-9, 1e-11)
    assert solver.stable_timestep() == 2e-3
    assert solver.last_dt_theta == 1e-9
    assert solver.last_dt_diff_explicit == 1e-11


def test_low_re_auto_step_refines_full_implicit_residual():
    solver = FlowSolver(SolverConfig(nr=40, ntheta=64, re=10, alpha=5))
    _, old_theta_limit, _ = solver.timestep_limits()
    dt = solver.stable_timestep()
    assert dt > 100.0 * old_theta_limit

    diagnostics = solver.step()

    assert np.isfinite(solver.omega).all()
    assert diagnostics["implicit_residual"] < 1e-5
    assert diagnostics["implicit_operator_applications"] > 0
