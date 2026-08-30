"""Pressure-Poisson recovery and surface force coefficients."""

from __future__ import annotations
import numpy as np

from .mapping import AirfoilMapping


def kutta_joukowski_circulation(
    mapping: AirfoilMapping, u_inf: float, alpha_deg: float
) -> float:
    """Return the clockwise-positive circulation selected by the Kutta condition.

    The rear stagnation point is placed on the generating-circle ray that
    points toward the positive Joukowski critical point.  Circulation is
    invariant under the conformal map, while the auxiliary free-stream speed
    is reduced by the chord-normalization scale.
    """
    if u_inf <= 0:
        raise ValueError("U_inf must be positive")
    theta_te = float(np.angle(complex(mapping.a) - mapping.center))
    alpha = np.deg2rad(alpha_deg)
    u_aux = u_inf / mapping.raw_chord
    return float(
        4.0
        * np.pi
        * mapping.circle_radius
        * u_aux
        * np.sin(alpha - theta_te)
    )


def analytical_kutta_joukowski_cp(
    mapping: AirfoilMapping,
    theta: np.ndarray,
    u_inf: float,
    alpha_deg: float,
) -> np.ndarray:
    """Analytical inviscid surface ``Cp`` on the mapped generating circle.

    This is the Bernoulli pressure profile obtained from uniform flow, its
    circle-theorem doublet, and the Kutta circulation.  It uses the same
    regularized Joukowski derivative as the viscous grid, so it is a
    geometry-aligned comparison curve and does not prescribe circulation in
    the Navier--Stokes solution.
    """
    if u_inf <= 0:
        raise ValueError("U_inf must be positive")

    theta = np.asarray(theta, dtype=float)
    zeta = mapping.zeta(np.zeros_like(theta), theta)
    circle_coordinate = zeta - mapping.center
    alpha = np.deg2rad(alpha_deg)
    u_aux = u_inf / mapping.raw_chord
    gamma = kutta_joukowski_circulation(mapping, u_inf, alpha_deg)

    dpotential_dzeta = u_aux * (
        np.exp(-1j * alpha)
        - np.exp(1j * alpha)
        * mapping.circle_radius**2
        / circle_coordinate**2
    ) + 1j * gamma / (2.0 * np.pi * circle_coordinate)
    complex_velocity = dpotential_dzeta / mapping.dz_dzeta(zeta)
    cp = 1.0 - np.abs(complex_velocity / u_inf) ** 2
    if not np.all(np.isfinite(cp)):
        raise FloatingPointError("analytical Kutta-Joukowski Cp is not finite")
    return np.ascontiguousarray(cp, dtype=float)


def scalar_gradient(field: np.ndarray, grid, H: np.ndarray,
                    es: np.ndarray, et: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fs = grid.s_derivative(field)
    ft = grid.theta_derivative(field)
    return ((fs * es.real + ft * et.real) / H,
            (fs * es.imag + ft * et.imag) / H)


def recover_pressure(solver) -> np.ndarray:
    u, v, _, _ = solver.velocity()
    ux, uy = scalar_gradient(u, solver.grid, solver.H, solver.es, solver.et)
    vx, vy = scalar_gradient(v, solver.grid, solver.H, solver.es, solver.et)
    source = -(ux * ux + 2.0 * uy * vx + vy * vy)
    p = solver.poisson.solve_neumann_wall(solver.H2 * source, outer=0.0)
    return np.ascontiguousarray(p)


def surface_coefficients(solver, pressure: np.ndarray) -> dict[str, float | np.ndarray]:
    qinf = 0.5 * solver.config.u_inf**2
    cp = pressure[0] / max(qinf, 1e-15)
    cp_kj = analytical_kutta_joukowski_cp(
        solver.mapping,
        solver.grid.theta,
        solver.config.u_inf,
        solver.config.alpha,
    )
    # Surface tangent length is H dtheta and body outward normal is +e_s.
    dtheta = 2 * np.pi / solver.config.ntheta
    force = -np.sum(pressure[0] * solver.es[0] * solver.H[0]) * dtheta
    alpha = np.deg2rad(solver.config.alpha)
    drag_dir = complex(np.cos(alpha), np.sin(alpha))
    lift_dir = complex(-np.sin(alpha), np.cos(alpha))
    cd = float((force.real * drag_dir.real + force.imag * drag_dir.imag) / max(qinf, 1e-15))
    cl = float((force.real * lift_dir.real + force.imag * lift_dir.imag) / max(qinf, 1e-15))
    return {
        "cp": cp,
        "cp_kj": cp_kj,
        "cl_pressure": cl,
        "cd_pressure": cd,
    }
