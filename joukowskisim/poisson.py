"""Fourier-modal Chebyshev Poisson and Helmholtz solves."""

from __future__ import annotations

import numpy as np
from scipy import fft, linalg

from .spectral import SpectralGrid


class PoissonSolver:
    def __init__(self, grid: SpectralGrid):
        self.grid = grid
        self._lu: list[tuple[np.ndarray, np.ndarray]] = []
        self._lu_neumann: list[tuple[np.ndarray, np.ndarray]] = []
        for kval in grid.kr.astype(int):
            A = grid.Dss.copy()
            A -= (kval * kval) * np.eye(grid.nr)
            A[0, :] = 0; A[0, 0] = 1
            A[-1, :] = 0; A[-1, -1] = 1
            self._lu.append(linalg.lu_factor(A, check_finite=False))
            A[0, :] = grid.Ds[0, :]
            self._lu_neumann.append(linalg.lu_factor(A, check_finite=False))

    def solve(self, rhs: np.ndarray, wall: float | np.ndarray = 0.0,
              outer: float | np.ndarray = 0.0) -> np.ndarray:
        b = fft.rfft(np.asarray(rhs), axis=-1, workers=-1)
        wh = fft.rfft(np.broadcast_to(wall, (self.grid.ntheta,)), workers=-1)
        oh = fft.rfft(np.broadcast_to(outer, (self.grid.ntheta,)), workers=-1)
        b[0, :] = wh
        b[-1, :] = oh
        result = np.empty_like(b)
        for m, lu in enumerate(self._lu):
            result[:, m] = linalg.lu_solve(lu, b[:, m], check_finite=False)
        return np.ascontiguousarray(fft.irfft(result, n=self.grid.ntheta, axis=-1, workers=-1))

    def solve_neumann_wall(self, rhs: np.ndarray, wall_derivative: float | np.ndarray = 0.0,
                           outer: float | np.ndarray = 0.0) -> np.ndarray:
        b = fft.rfft(np.asarray(rhs), axis=-1, workers=-1)
        wh = fft.rfft(np.broadcast_to(wall_derivative, (self.grid.ntheta,)), workers=-1)
        oh = fft.rfft(np.broadcast_to(outer, (self.grid.ntheta,)), workers=-1)
        b[0, :] = wh; b[-1, :] = oh
        result = np.empty_like(b)
        for m, lu in enumerate(self._lu_neumann):
            result[:, m] = linalg.lu_solve(lu, b[:, m], check_finite=False)
        return np.ascontiguousarray(fft.irfft(result, n=self.grid.ntheta, axis=-1, workers=-1))

