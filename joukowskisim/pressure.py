"""Pressure-Poisson recovery and surface force coefficients."""

from __future__ import annotations
import numpy as np


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
    # Surface tangent length is H dtheta and body outward normal is +e_s.
    dtheta = 2 * np.pi / solver.config.ntheta
    force = -np.sum(pressure[0] * solver.es[0] * solver.H[0]) * dtheta
    alpha = np.deg2rad(solver.config.alpha)
    drag_dir = complex(np.cos(alpha), np.sin(alpha))
    lift_dir = complex(-np.sin(alpha), np.cos(alpha))
    cd = float((force.real * drag_dir.real + force.imag * drag_dir.imag) / max(qinf, 1e-15))
    cl = float((force.real * lift_dir.real + force.imag * lift_dir.imag) / max(qinf, 1e-15))
    return {"cp": cp, "cl_pressure": cl, "cd_pressure": cd}

