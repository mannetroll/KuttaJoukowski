"""Fourier-modal Chebyshev Poisson and Helmholtz solves."""

from __future__ import annotations

import numpy as np
from scipy import fft, linalg

from .batched_lu import solve_cached_columns
from .spectral import SpectralGrid


class PoissonSolver:
    def __init__(self, grid: SpectralGrid):
        self.grid = grid
        self._lu: list[tuple[np.ndarray, np.ndarray]] = []
        self._lu_neumann: list[tuple[np.ndarray, np.ndarray]] = []
        self._near_wall_inverse_rows: np.ndarray | None = None
        self._near_wall_functional_rows: dict[
            tuple[float, float], np.ndarray
        ] = {}
        for kval in grid.kr.astype(int):
            A = grid.Dss.copy()
            A -= (kval * kval) * np.eye(grid.nr)
            A[0, :] = 0; A[0, 0] = 1
            A[-1, :] = 0; A[-1, -1] = 1
            self._lu.append(linalg.lu_factor(A, check_finite=False))

    def _ensure_near_wall_inverse_rows(self) -> None:
        """Cache rows 1 and 2 of every homogeneous Poisson inverse.

        A full implicit diffusion Krylov product needs only the generalized
        Thom functional of the streamfunction response, not the complete
        streamfunction.  Two transposed solves per Fourier mode turn that
        operation into FFT + small dense contractions thereafter.
        """
        if self._near_wall_inverse_rows is not None:
            return
        probes = np.zeros((self.grid.nr, 2), dtype=float)
        probes[1, 0] = 1.0
        probes[2, 1] = 1.0
        rows = np.empty(
            (len(self._lu), 2, self.grid.nr),
            dtype=float,
        )
        for mode, factor in enumerate(self._lu):
            dual = linalg.lu_solve(
                factor,
                probes,
                trans=1,
                check_finite=False,
            )
            rows[mode] = dual.T
        self._near_wall_inverse_rows = np.ascontiguousarray(rows)

    def solve_homogeneous_near_wall(self, rhs: np.ndarray) -> np.ndarray:
        """Return radial rows 1 and 2 of a homogeneous Poisson solve."""
        source = np.asarray(rhs)
        if source.shape != (self.grid.nr, self.grid.ntheta):
            raise ValueError("Poisson RHS has the wrong grid shape")
        self._ensure_near_wall_inverse_rows()
        transformed = fft.rfft(source, axis=-1, workers=-1)
        transformed[[0, -1], :] = 0.0
        samples_hat = np.einsum(
            "kmr,rk->mk",
            self._near_wall_inverse_rows,
            transformed,
            # This fixed two-row contraction is faster without NumPy's
            # per-call path optimizer in the Krylov hot path.
            optimize=False,
        )
        return np.ascontiguousarray(
            fft.irfft(
                samples_hat,
                n=self.grid.ntheta,
                axis=-1,
                workers=-1,
            )
        )

    def solve_homogeneous_near_wall_functional(
        self,
        rhs: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Apply a fixed linear functional to homogeneous near-wall rows.

        Combining the two cached inverse rows before the modal contraction is
        algebraically identical to combining their physical-space solutions,
        but needs only one radial contraction and one inverse FFT.
        """
        source = np.asarray(rhs)
        if source.shape != (self.grid.nr, self.grid.ntheta):
            raise ValueError("Poisson RHS has the wrong grid shape")
        coefficients = np.asarray(weights, dtype=float)
        if coefficients.shape != (2,):
            raise ValueError("near-wall functional requires two weights")
        self._ensure_near_wall_inverse_rows()
        key = (float(coefficients[0]), float(coefficients[1]))
        functional_rows = self._near_wall_functional_rows.get(key)
        if functional_rows is None:
            functional_rows = np.ascontiguousarray(
                np.einsum(
                    "m,kmr->kr",
                    coefficients,
                    self._near_wall_inverse_rows,
                    optimize=False,
                )
            )
            self._near_wall_functional_rows[key] = functional_rows

        transformed = fft.rfft(source, axis=-1, workers=-1)
        transformed[[0, -1], :] = 0.0
        samples_hat = np.einsum(
            "kr,rk->k",
            functional_rows,
            transformed,
            optimize=False,
        )
        return np.ascontiguousarray(
            fft.irfft(
                samples_hat,
                n=self.grid.ntheta,
                workers=-1,
            )
        )

    def solve_modes_homogeneous_near_wall(
        self,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """Return rows 1 and 2 for every mode and a batch of RHS columns.

        ``rhs`` has radial rows on its leading axis and independent source
        columns on its trailing axis.  Unlike
        :meth:`solve_homogeneous_near_wall`, this method deliberately does
        not Fourier transform those columns: callers such as the Thom Schur
        assembly are already constructing the response of each Fourier mode
        to each physical-theta basis column explicitly.

        The returned shape is ``(rfft_modes, 2, rhs_columns)``.  Homogeneous
        radial boundary data are imposed by ignoring the first and last RHS
        rows, exactly as ``solve_mode(mode, rhs)`` does with its default
        boundary arguments.
        """
        source = np.asarray(rhs)
        if source.ndim != 2 or source.shape[0] != self.grid.nr:
            raise ValueError("modal Poisson RHS batch has the wrong shape")
        self._ensure_near_wall_inverse_rows()
        inverse_rows = self._near_wall_inverse_rows[:, :, 1:-1]
        mode_count = inverse_rows.shape[0]
        dtype = np.result_type(source.dtype, np.float64)
        interior_source = np.asarray(source[1:-1], dtype=dtype)
        samples = (
            inverse_rows.reshape(2 * mode_count, self.grid.nr - 2)
            @ interior_source
        )
        return np.ascontiguousarray(
            samples.reshape(mode_count, 2, source.shape[1])
        )

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
        result = solve_cached_columns(self._lu, b)
        return np.ascontiguousarray(fft.irfft(result, n=self.grid.ntheta, axis=-1, workers=-1))

    def solve_neumann_wall(self, rhs: np.ndarray, wall_derivative: float | np.ndarray = 0.0,
                           outer: float | np.ndarray = 0.0) -> np.ndarray:
        self._ensure_neumann_factors()
        b = fft.rfft(np.asarray(rhs), axis=-1, workers=-1)
        wh = fft.rfft(np.broadcast_to(wall_derivative, (self.grid.ntheta,)), workers=-1)
        oh = fft.rfft(np.broadcast_to(outer, (self.grid.ntheta,)), workers=-1)
        b[0, :] = wh; b[-1, :] = oh
        result = solve_cached_columns(self._lu_neumann, b)
        return np.ascontiguousarray(fft.irfft(result, n=self.grid.ntheta, axis=-1, workers=-1))
