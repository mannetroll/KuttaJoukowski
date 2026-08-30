"""Integral and pointwise flow diagnostics."""

from __future__ import annotations
import numpy as np


def circulation_from_streamfunction(psi_s: np.ndarray, radial_index: int = 2) -> float:
    """Clockwise-positive circulation, so positive alpha normally gives +Cl."""
    return float(np.mean(psi_s[radial_index]) * 2 * np.pi)


def kinetic_energy(qs: np.ndarray, qt: np.ndarray, H2: np.ndarray,
                   s: np.ndarray) -> float:
    angular = np.mean((qs * qs + qt * qt) * H2, axis=1) * 2 * np.pi
    return float(0.5 * abs(np.trapezoid(angular, s)))


def divergence(u: np.ndarray, v: np.ndarray, grid, mapping) -> np.ndarray:
    us = grid.s_derivative(u); ut = grid.theta_derivative(u)
    vs = grid.s_derivative(v); vt = grid.theta_derivative(v)
    S, T = np.meshgrid(grid.s, grid.theta, indexing="ij")
    es, et = mapping.coordinate_basis(S, T)
    H = mapping.metric(S, T)
    ux = (us * es.real + ut * et.real) / H
    vy = (vs * es.imag + vt * et.imag) / H
    return ux + vy

