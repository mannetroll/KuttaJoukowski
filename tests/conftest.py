import pytest
from joukowskisim.solver import FlowSolver, SolverConfig


@pytest.fixture(scope="module")
def small_solver():
    return FlowSolver(SolverConfig(nr=20, ntheta=48, re=200, alpha=5))
