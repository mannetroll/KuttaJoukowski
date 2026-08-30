import numpy as np
from joukowskisim.spectral import SpectralGrid
from joukowskisim.poisson import PoissonSolver


def test_poisson_manufactured_solution():
    g=SpectralGrid(26,48,2.5); p=PoissonSolver(g)
    S,T=np.meshgrid(g.s,g.theta,indexing='ij')
    exact=S*(g.s_max-S)*np.cos(3*T)
    rhs=(-2-9*S*(g.s_max-S))*np.cos(3*T)
    got=p.solve(rhs,exact[0],exact[-1])
    assert np.max(abs(got-exact))<2e-10


def test_pressure_neumann_poisson_manufactured():
    g=SpectralGrid(24,40,2.0); p=PoissonSolver(g)
    assert not p._lu_neumann
    S,T=np.meshgrid(g.s,g.theta,indexing='ij')
    exact=(S**2-g.s_max**2)*np.cos(2*T)
    rhs=(2-4*(S**2-g.s_max**2))*np.cos(2*T)
    got=p.solve_neumann_wall(rhs,wall_derivative=np.zeros(g.ntheta),outer=np.zeros(g.ntheta))
    assert len(p._lu_neumann) == g.ntheta // 2 + 1
    assert np.max(abs(got-exact))<2e-9


def test_batched_modal_near_wall_rows_match_complete_mode_solves():
    grid = SpectralGrid(18, 32, 2.25)
    poisson = PoissonSolver(grid)
    rng = np.random.default_rng(4815)
    real_source = rng.standard_normal((grid.nr, 11))
    # Homogeneous mode solves replace both boundary rows; nonzero values here
    # verify that the cached-row contraction follows the same convention.
    real_source[[0, -1]] = rng.standard_normal((2, 11))

    sources = (
        real_source,
        real_source + 1j * rng.standard_normal(real_source.shape),
    )
    for source in sources:
        expected = np.stack(
            [
                poisson.solve_mode(mode, source)[[1, 2]]
                for mode in range(grid.ntheta // 2 + 1)
            ]
        )
        got = poisson.solve_modes_homogeneous_near_wall(source)
        assert got.shape == (grid.ntheta // 2 + 1, 2, source.shape[1])
        assert np.iscomplexobj(got) == np.iscomplexobj(source)
        assert np.allclose(got, expected, rtol=8e-13, atol=8e-13)
