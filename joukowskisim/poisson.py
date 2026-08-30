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

    def _ensure_neumann_factors(self) -> None:
        """Build pressure-only Neumann factors lazily on first pressure solve."""
        if self._lu_neumann:
            return
        for kval in self.grid.kr.astype(int):
            matrix = self.grid.Dss.copy()
            matrix -= (kval * kval) * np.eye(self.grid.nr)
            matrix[0, :] = self.grid.Ds[0, :]
            matrix[-1, :] = 0.0
            matrix[-1, -1] = 1.0
            self._lu_neumann.append(
                linalg.lu_factor(matrix, check_finite=False)
            )

    def solve_mode(
        self,
        mode: int,
        rhs: np.ndarray,
        wall: float | np.ndarray = 0.0,
        outer: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Solve one Fourier mode, optionally for a batch of right-hand sides.

        The leading axis of ``rhs`` is radial.  Any trailing axes are treated
        as independent right-hand sides by LAPACK.  This is used to build the
        cached wall-influence Schur matrices without repeating FFTs.
        """
        if mode < 0 or mode >= len(self._lu):
            raise ValueError("Fourier mode is outside the rFFT range")
        source = np.asarray(rhs)
        if source.ndim < 1 or source.shape[0] != self.grid.nr:
            raise ValueError("modal Poisson RHS has the wrong radial size")
        dtype = np.result_type(source.dtype, np.float64)
        b = np.array(source, dtype=dtype, copy=True)
        b[0] = wall
        b[-1] = outer
        result = linalg.lu_solve(
            self._lu[mode],
            b,
            check_finite=False,
        )
        return np.ascontiguousarray(result)

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
        self._ensure_neumann_factors()
        b = fft.rfft(np.asarray(rhs), axis=-1, workers=-1)
        wh = fft.rfft(np.broadcast_to(wall_derivative, (self.grid.ntheta,)), workers=-1)
        oh = fft.rfft(np.broadcast_to(outer, (self.grid.ntheta,)), workers=-1)
        b[0, :] = wh; b[-1, :] = oh
        result = np.empty_like(b)
        for m, lu in enumerate(self._lu_neumann):
            result[:, m] = linalg.lu_solve(lu, b[:, m], check_finite=False)
        return np.ascontiguousarray(fft.irfft(result, n=self.grid.ntheta, axis=-1, workers=-1))
