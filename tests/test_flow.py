import numpy as np
from joukowskisim.boundary import update_wall_vorticity, wall_velocity_error
from joukowskisim.diagnostics import divergence
from joukowskisim.solver import FlowSolver, SolverConfig


def test_uniform_flow_velocity_recovery_and_incompressibility(small_solver):
    s=small_solver; a=np.deg2rad(13); U=1.3
    psi=U*(np.cos(a)*s.y-np.sin(a)*s.x)
    u,v,_,_=s.velocity(psi)
    core=np.s_[1:-1,:]
    assert np.max(abs(u[core]-U*np.cos(a)))<2e-9
    assert np.max(abs(v[core]-U*np.sin(a)))<2e-9
    assert np.max(abs(divergence(u,v,s.grid,s.mapping)[2:-2]))<2e-7


def test_wall_impermeability_and_no_slip_relation(small_solver):
    s=small_solver; S=s.S
    psi=s.wall_psi_target + S**2*np.sin(s.T)
    omega=np.zeros_like(psi); update_wall_vorticity(omega,psi,s.grid.Dss,s.H2,s.grid.s)
    assert np.max(abs(s.grid.theta_derivative(psi)[0]/s.H[0]))<1e-12
    assert wall_velocity_error(psi,s.grid.Ds,s.H)<2e-10
    assert np.isfinite(omega).all() and np.max(abs(omega[0]))>0


def test_far_field_velocity_approaches_uniform(small_solver):
    u,v,_,_=small_solver.velocity(); a=np.deg2rad(small_solver.config.alpha)
    # Tangential differentiation of the exact outer psi fixes its normal component.
    normal=(u[-1]*small_solver.es[-1].real+v[-1]*small_solver.es[-1].imag)
    target=small_solver.config.u_inf*(np.cos(a)*small_solver.es[-1].real+np.sin(a)*small_solver.es[-1].imag)
    assert np.sqrt(np.mean((normal-target)**2))<2e-8


def test_deterministic_initialization():
    c=SolverConfig(nr=14,ntheta=32,re=100,startup_time=0)
    a=FlowSolver(c); b=FlowSolver(c)
    assert np.array_equal(a.omega,b.omega) and np.array_equal(a.psi,b.psi)


def test_positive_finite_timestep_and_trailing_edge_stability(small_solver):
    dt=small_solver.stable_timestep()
    assert np.isfinite(dt) and 1e-12<dt<=.02


def test_finite_values_after_timesteps():
    s=FlowSolver(SolverConfig(nr=16,ntheta=32,re=100,alpha=8,startup_time=.01))
    for _ in range(12): s.step()
    assert np.isfinite(s.omega).all() and np.isfinite(s.psi).all()
    assert s.step_count==12 and s.diagnostics()['max_omega']>0


def test_viscous_decay_operator():
    s=FlowSolver(SolverConfig(nr=16,ntheta=32,re=50,startup_time=0))
    w=np.sin(np.pi*s.S/s.mapping.s_max)*np.sin(2*s.T)
    diffusion=s.config.nu*s.grid.laplacian_coordinate(w)/s.H2
    weighted=np.sum(w*diffusion*s.H2)
    assert weighted<0


def test_pressure_cp_finite(small_solver):
    from joukowskisim.pressure import surface_coefficients
    p=small_solver.compute_pressure(); c=surface_coefficients(small_solver,p)
    assert np.isfinite(p).all() and np.isfinite(c['cp']).all()
    assert np.isfinite(c['cl_pressure']) and np.isfinite(c['cd_pressure'])


def test_headless_wake_and_symmetry_sanity():
    s=FlowSolver(SolverConfig(nr=16,ntheta=40,re=150,alpha=10,startup_time=.002))
    for _ in range(15): s.step()
    downstream=(s.x>1.0) & (s.x<3.0)
    assert np.max(abs(s.omega[downstream]))>1e-12
    z=FlowSolver(SolverConfig(nr=16,ntheta=40,re=150,alpha=0,startup_time=.002))
    for _ in range(8): z.step()
    # Reflection symmetry: vorticity changes sign across the chord plane.
    reflected=np.roll(z.omega[:,::-1],1,axis=1)
    scale=max(np.max(abs(z.omega)),1e-12)
    assert np.max(abs(z.omega+reflected))/scale<.15


def test_circulation_sign_convention_is_documented_and_consistent():
    from joukowskisim.diagnostics import circulation_from_streamfunction
    # Clockwise-positive Gamma is defined as integral psi_s dtheta.
    psi_s=np.ones((5,32))*0.25
    assert circulation_from_streamfunction(psi_s,2)>0

