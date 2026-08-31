import numpy as np
from joukowskisim.boundary import update_wall_vorticity, wall_velocity_error
from joukowskisim.diagnostics import divergence
from joukowskisim.pressure import (
    analytical_kutta_joukowski_cp,
    kutta_joukowski_circulation,
    surface_coefficients,
)
from joukowskisim.solver import FlowSolver, SolverConfig


def test_uniform_flow_velocity_recovery_and_incompressibility(small_solver):
    s=small_solver; a=np.deg2rad(13); U=1.3
    psi=U*(np.cos(a)*s.y-np.sin(a)*s.x)
    u,v,_,_=s.velocity(psi)
    core=np.s_[1:-1,:]
    assert np.max(abs(u[core]-U*np.cos(a)))<2e-9
    assert np.max(abs(v[core]-U*np.sin(a)))<2e-9
    assert np.max(abs(divergence(u,v,s.grid,s.mapping)[2:-2]))<2e-7


def test_nonlinear_term_advects_vorticity_with_recovered_velocity(small_solver):
    # For psi=y, the recovered velocity is (u,v)=(1,0).  With omega=x,
    # u dot grad(omega) must therefore be +1, so the evolution RHS uses -1.
    s=small_solver
    advection=s.nonlinear_term(s.x,s.y)
    core=np.s_[2:-3,:]
    np.testing.assert_allclose(advection[core],1.0,rtol=1e-8,atol=1e-8)
    evolution=s._explicit_rhs(s.x,s.y)
    no_sponge=(s.S>0.0)&(s.S<0.8*s.mapping.s_max)
    np.testing.assert_allclose(evolution[no_sponge],-1.0,rtol=1e-8,atol=1e-8)


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
    c=SolverConfig(nr=14,ntheta=32,re=100)
    a=FlowSolver(c); b=FlowSolver(c)
    assert np.array_equal(a.omega,b.omega) and np.array_equal(a.psi,b.psi)


def test_positive_finite_timestep_and_trailing_edge_stability(small_solver):
    dt=small_solver.stable_timestep()
    assert np.isfinite(dt) and 1e-12<dt<=.02


def test_finite_values_after_timesteps():
    s=FlowSolver(SolverConfig(nr=16,ntheta=32,re=100,alpha=8))
    for _ in range(12): s.step()
    assert np.isfinite(s.omega).all() and np.isfinite(s.psi).all()
    assert s.step_count==12 and s.diagnostics()['max_omega']>0


def test_viscous_decay_operator():
    s=FlowSolver(SolverConfig(nr=16,ntheta=32,re=50))
    w=np.sin(np.pi*s.S/s.mapping.s_max)*np.sin(2*s.T)
    diffusion=s.config.nu*s.grid.laplacian_coordinate(w)/s.H2
    weighted=np.sum(w*diffusion*s.H2)
    assert weighted<0


def test_pressure_cp_finite(small_solver):
    p=small_solver.compute_pressure(); c=surface_coefficients(small_solver,p)
    assert np.isfinite(p).all() and np.isfinite(c['cp']).all()
    assert c['cp_kj'].shape == (small_solver.config.ntheta,)
    assert np.isfinite(c['cp_kj']).all()
    assert np.isfinite(c['cl_pressure']) and np.isfinite(c['cd_pressure'])


def test_analytical_kutta_joukowski_cp_properties():
    s=FlowSolver(SolverConfig(nr=12,ntheta=128,re=100,alpha=7))
    cp=analytical_kutta_joukowski_cp(
        s.mapping, s.grid.theta, s.config.u_inf, s.config.alpha
    )
    cp_fast=analytical_kutta_joukowski_cp(
        s.mapping, s.grid.theta, 2.5*s.config.u_inf, s.config.alpha
    )
    assert cp.shape == (s.config.ntheta,)
    assert np.isfinite(cp).all() and np.max(cp) <= 1.0 + 1e-12
    assert np.isclose(cp[0], 1.0, atol=1e-12)
    assert np.allclose(cp, cp_fast, rtol=2e-13, atol=2e-13)
    assert kutta_joukowski_circulation(
        s.mapping, s.config.u_inf, s.config.alpha
    ) > 0

    symmetric=analytical_kutta_joukowski_cp(
        s.mapping, s.grid.theta, s.config.u_inf, 0.0
    )
    assert np.allclose(symmetric, np.roll(symmetric[::-1], 1), atol=2e-12)


def test_analytical_pressure_integrates_to_kutta_joukowski_lift():
    s=FlowSolver(SolverConfig(
        nr=12,ntheta=128,re=100,alpha=7,camber=.03
    ))
    cp=analytical_kutta_joukowski_cp(
        s.mapping, s.grid.theta, s.config.u_inf, s.config.alpha
    )
    qinf=0.5*s.config.u_inf**2
    force=-np.sum(qinf*cp*s.es[0]*s.H[0])*(2*np.pi/s.config.ntheta)
    alpha=np.deg2rad(s.config.alpha)
    cd=(force.real*np.cos(alpha)+force.imag*np.sin(alpha))/qinf
    cl=(-force.real*np.sin(alpha)+force.imag*np.cos(alpha))/qinf
    expected_cl=2*kutta_joukowski_circulation(
        s.mapping, s.config.u_inf, s.config.alpha
    )/s.config.u_inf
    assert abs(cd) < 5e-4
    assert np.isclose(cl, expected_cl, rtol=5e-4, atol=5e-4)


def test_headless_wake_and_symmetry_sanity():
    s=FlowSolver(SolverConfig(nr=16,ntheta=40,re=150,alpha=10))
    for _ in range(15): s.step()
    downstream=(s.x>1.0) & (s.x<3.0)
    assert np.max(abs(s.omega[downstream]))>1e-12
    z=FlowSolver(SolverConfig(nr=16,ntheta=40,re=150,alpha=0))
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
