"""Low-storage IMEX-RK3 support for stiff mapped diffusion."""

from __future__ import annotations

import time
import numpy as np
from scipy import fft, linalg

from .batched_lu import solve_cached_columns


# Spalart/Moser/Rogers low-storage coefficients, matching the reference
# implementation in turbo_simulator.py.
LS_IMEX_ALPHA = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
LS_IMEX_BETA = (0.0, -17.0 / 60.0, -5.0 / 12.0)


class MappedImplicitDiffusion:
    """Complete mapped viscous operator treated by the implicit stages.

    Multiplication by ``H^-2`` does not commute with a Fourier transform, so
    azimuthal diffusion is not diagonal in Fourier space.  The production
    stage solve therefore uses this exact operator in a Krylov method, with
    :class:`RadialImplicitDiffusion` as its cached preconditioner.
    """

    def __init__(self, grid, h2: np.ndarray, nu: float):
        self.grid = grid
        self.h2 = np.asarray(h2, dtype=float)
        self.nu = float(nu)

    def apply(self, field: np.ndarray) -> np.ndarray:
        out = self.nu * (
            self.grid.Dss @ field
            + self.grid.theta_derivative(field, 2)
        ) / self.h2
        out[[0, -1], :] = 0.0
        return np.ascontiguousarray(out)


class RadialImplicitDiffusion:
    """Cached per-angle LU solves for ``nu * H^-2 * d_ss``.

    The conformal metric makes the radial operator angle-dependent, but it
    remains uncoupled between theta columns.  Three sets of factors are cached
    for the LS-IMEX-RK3 diagonal coefficients and rebuilt only when ``dt``
    changes.
    """

    def __init__(self, grid, h2: np.ndarray, nu: float):
        self.grid = grid
        self.h2 = np.asarray(h2, dtype=float)
        self.nu = float(nu)
        self._interior = slice(1, -1)
        self._dii = np.ascontiguousarray(grid.Dss[1:-1, 1:-1])
        self._wall_column = np.ascontiguousarray(grid.Dss[1:-1, 0])
        self._outer_column = np.ascontiguousarray(grid.Dss[1:-1, -1])
        self._coefficient = np.ascontiguousarray(self.nu / self.h2[1:-1])
        self._identity = np.eye(grid.nr - 2)
        self._dt: float | None = None
        self._factors: list[list[tuple[np.ndarray, np.ndarray]]] = []
        self._wall_responses: list[np.ndarray] = []
        self.factorizations = 0
        self.factor_seconds = 0.0
        self.solve_seconds = 0.0

    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply radial diffusion, leaving boundary evolution to the closure."""
        out = self.nu * (self.grid.Dss @ field) / self.h2
        out[[0, -1], :] = 0.0
        return np.ascontiguousarray(out)

    def prepare(self, dt: float) -> None:
        dt = float(dt)
        if self._dt == dt and self._factors:
            return
        started = time.perf_counter()
        # Drop the old O(Ntheta*Nr^2) cache before constructing its replacement
        # so an adaptive-dt rebuild does not require roughly twice the memory.
        self._factors = []
        self._wall_responses = []
        self._dt = None
        factors: list[list[tuple[np.ndarray, np.ndarray]]] = []
        wall_responses: list[np.ndarray] = []
        for alpha in LS_IMEX_ALPHA:
            stage_factors = []
            scale = alpha * dt
            response_rhs = (
                scale
                * self._coefficient
                * self._wall_column[:, None]
            )
            response = np.empty_like(response_rhs)
            for column in range(self.grid.ntheta):
                matrix = self._identity - scale * (
                    self._coefficient[:, column, None] * self._dii
                )
                factor = linalg.lu_factor(matrix, check_finite=False)
                stage_factors.append(factor)
                response[:, column] = linalg.lu_solve(
                    factor,
                    response_rhs[:, column],
                    check_finite=False,
                )
            factors.append(stage_factors)
            wall_responses.append(np.ascontiguousarray(response))
        self._factors = factors
        self._wall_responses = wall_responses
        self._dt = dt
        self.factorizations += 1
        self.factor_seconds += time.perf_counter() - started

    def wall_response(self, stage: int) -> np.ndarray:
        """Return the interior response to unit wall vorticity.

        For a prepared stage this is

        ``(I - alpha*dt*L_II)^-1 alpha*dt*L_I0``.

        It is diagonal in physical theta, so multiplying the returned array
        by a wall vector broadcasts the complete interior influence field.
        """
        if self._dt is None or not self._wall_responses:
            raise RuntimeError("implicit radial factors have not been prepared")
        return self._wall_responses[stage]

    def solve(
        self,
        rhs: np.ndarray,
        stage: int,
        wall: np.ndarray,
        outer: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve one implicit stage with supplied vorticity boundary guesses."""
        if self._dt is None or not self._factors:
            raise RuntimeError("implicit radial factors have not been prepared")
        started = time.perf_counter()
        alpha_dt = LS_IMEX_ALPHA[stage] * self._dt
        wall = np.asarray(wall, dtype=float)
        if outer is None:
            outer = np.zeros_like(wall)
        else:
            outer = np.asarray(outer, dtype=float)
        boundary_second_derivative = (
            self._wall_column[:, None] * wall[None, :]
            + self._outer_column[:, None] * outer[None, :]
        )
        interior_rhs = np.ascontiguousarray(
            rhs[1:-1]
            + alpha_dt * self._coefficient * boundary_second_derivative
        )
        interior = solve_cached_columns(
            self._factors[stage],
            interior_rhs,
        )
        out = np.empty_like(rhs)
        out[0] = wall
        out[1:-1] = interior
        out[-1] = outer
        self.solve_seconds += time.perf_counter() - started
        return np.ascontiguousarray(out)

    def residual(self, field: np.ndarray, rhs: np.ndarray, stage: int) -> float:
        """Return the relative interior residual of the implicit stage solve."""
        alpha_dt = LS_IMEX_ALPHA[stage] * float(self._dt or 0.0)
        defect = field - alpha_dt * self.apply(field) - rhs
        numerator = float(np.max(np.abs(defect[1:-1])))
        denominator = max(1.0, float(np.max(np.abs(rhs[1:-1]))))
        return numerator / denominator


class ThomWallInfluence:
    """Cached Schur complements for the local Thom wall closure.

    A radial implicit solve is diagonal in physical theta: its response to a
    wall-vorticity vector is ``R(s, theta) * omega_wall(theta)``.  The Poisson
    solve couples those columns in theta.  If ``B`` is the local cubic Thom
    operator, a stage must satisfy

    ``omega_wall = B(psi_base) + B(Poisson(R * omega_wall))``.

    This class assembles and factors the dense ``I - B P R`` system once per
    ``(dt, stage)``.  Unlike a direct wall-derivative projection, this system
    tends continuously to the identity as ``dt`` tends to zero.
    """

    def __init__(self, grid, h2: np.ndarray, poisson, radial):
        self.grid = grid
        self.h2 = np.asarray(h2, dtype=float)
        self.poisson = poisson
        self.radial = radial
        self._dt: float | None = None
        self._matrices: list[np.ndarray] = []
        self._factors: list[tuple[np.ndarray, np.ndarray]] = []
        self.conditions: tuple[float, ...] = ()
        self.assemblies = 0
        self.assembly_seconds = 0.0
        s1, s2 = float(grid.s[1]), float(grid.s[2])
        reconstruction = np.array(
            [[s1**2, s1**3], [s2**2, s2**3]],
            dtype=float,
        )
        self._wall_weights = 2.0 * np.linalg.inv(reconstruction)[0]

    def _assemble_stage(self, stage: int) -> np.ndarray:
        response = self.radial.wall_response(stage)
        source = np.zeros((self.grid.nr, self.grid.ntheta), dtype=float)
        source[1:-1] = -self.h2[1:-1] * response

        mode_count = self.grid.ntheta // 2 + 1
        modal_curvature = np.empty(
            (mode_count, self.grid.ntheta),
            dtype=np.complex128,
        )
        for mode in range(mode_count):
            streamfunction = self.poisson.solve_mode(mode, source)
            modal_curvature[mode] = -(
                self._wall_weights[0] * streamfunction[1]
                + self._wall_weights[1] * streamfunction[2]
            )

        modes = np.arange(mode_count, dtype=float)[:, None]
        columns = np.arange(self.grid.ntheta, dtype=float)[None, :]
        basis_phase = np.exp(
            -2j * np.pi * modes * columns / self.grid.ntheta
        )
        closure_response = fft.irfft(
            modal_curvature * basis_phase,
            n=self.grid.ntheta,
            axis=0,
            workers=-1,
        )
        closure_response /= self.h2[0, :, None]
        matrix = np.eye(self.grid.ntheta) - closure_response.real
        return np.ascontiguousarray(matrix)

    def prepare(self, dt: float) -> None:
        dt = float(dt)
        if self._dt == dt and self._factors:
            return
        if self.radial._dt != dt:  # Both caches must describe the same stage.
            raise RuntimeError("radial implicit factors must be prepared first")
        started = time.perf_counter()
        self._matrices = []
        self._factors = []
        self._dt = None
        matrices = []
        factors = []
        conditions = []
        for stage in range(len(LS_IMEX_ALPHA)):
            matrix = self._assemble_stage(stage)
            factor = linalg.lu_factor(matrix, check_finite=False)
            matrix_norm = float(linalg.norm(matrix, 1, check_finite=False))
            gecon = linalg.lapack.get_lapack_funcs("gecon", (factor[0],))
            reciprocal_condition, info = gecon(
                factor[0], matrix_norm, norm="1"
            )
            condition = (
                float(1.0 / reciprocal_condition)
                if info == 0 and reciprocal_condition > 0.0
                else np.inf
            )
            if not np.isfinite(condition) or condition > 1e10:
                raise FloatingPointError(
                    "Thom wall influence matrix is ill-conditioned "
                    f"at stage {stage} (condition {condition:.3e})"
                )
            matrices.append(matrix)
            factors.append(factor)
            conditions.append(condition)
        self._matrices = matrices
        self._factors = factors
        self.conditions = tuple(conditions)
        self._dt = dt
        self.assemblies += 1
        self.assembly_seconds += time.perf_counter() - started

    def solve(self, stage: int, closure_rhs: np.ndarray) -> np.ndarray:
        """Solve the cached Thom Schur system for wall vorticity."""
        if self._dt is None or not self._factors:
            raise RuntimeError("wall influence factors have not been prepared")
        result = linalg.lu_solve(
            self._factors[stage],
            np.asarray(closure_rhs, dtype=float),
            check_finite=False,
        )
        return np.ascontiguousarray(result)

    def apply(self, stage: int, wall: np.ndarray) -> np.ndarray:
        """Apply a cached Thom Schur matrix (primarily for verification)."""
        if self._dt is None or not self._matrices:
            raise RuntimeError("wall influence factors have not been prepared")
        return np.ascontiguousarray(
            self._matrices[stage] @ np.asarray(wall, dtype=float)
        )

    def residual(
        self,
        stage: int,
        wall: np.ndarray,
        closure_rhs: np.ndarray,
    ) -> float:
        """Return the relative residual of the dense wall Schur solve."""
        defect = self._matrices[stage] @ wall - closure_rhs
        numerator = float(np.max(np.abs(defect)))
        denominator = max(1.0, float(np.max(np.abs(closure_rhs))))
        return numerator / denominator
