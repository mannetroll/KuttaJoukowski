"""Vorticity boundary conditions for the body-fitted wall."""

from __future__ import annotations
import numpy as np


def update_wall_vorticity(omega: np.ndarray, psi: np.ndarray,
                          Dss: np.ndarray, H2: np.ndarray,
                          s: np.ndarray | None = None) -> None:
    """Apply a global spectral Thom relation at the no-slip wall.

    Since psi is constant on s=0, psi_tt=0 there and -omega =
    psi_ss/H^2.  The Chebyshev second-derivative boundary row uses every
    radial point and is the spectral generalization of the local Thom formula.
    """
    if s is None:
        psi_ss_wall = Dss[0, :] @ psi
    else:
        # Cubic one-sided reconstruction with psi_s(0)=0.  Unlike merely
        # evaluating the Poisson equation at a Dirichlet wall, this supplies
        # the missing no-slip condition through the vorticity boundary value.
        s1, s2 = float(s[1]), float(s[2])
        M = np.array([[s1**2, s1**3], [s2**2, s2**3]])
        weights = 2.0 * np.linalg.inv(M)[0]
        psi_ss_wall = weights[0] * (psi[1] - psi[0]) + weights[1] * (psi[2] - psi[0])
    omega[0, :] = -psi_ss_wall / H2[0, :]
    omega[-1, :] = 0.0


def wall_velocity_error(psi: np.ndarray, Ds: np.ndarray, H: np.ndarray) -> float:
    # No penetration follows from constant psi; this measures tangential slip.
    return float(np.max(np.abs((Ds[0, :] @ psi) / H[0, :])))
