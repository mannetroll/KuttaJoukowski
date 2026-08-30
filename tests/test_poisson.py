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
