"""Low-storage IMEX-RK3 support for stiff mapped diffusion."""

from __future__ import annotations

import time
import numpy as np
from scipy import fft, linalg

from .batched_lu import solve_cached_groups


# Spalart/Moser/Rogers low-storage coefficients, matching the reference
# implementation in turbo_simulator.py.
LS_IMEX_ALPHA = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
LS_IMEX_BETA = (0.0, -17.0 / 60.0, -5.0 / 12.0)

# This approximation is confined to the radial Krylov preconditioner.  The
# production matrix-vector product and its independently evaluated residual
# continue to use the complete, ungrouped mapped diffusion operator.
RADIAL_PRECONDITIONER_PROFILE_TOLERANCE = 0.10
_EXACT_MIRROR_TOLERANCE = 512.0 * np.finfo(float).eps


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
    """Cached approximate radial preconditioner for mapped diffusion.

    The conformal metric makes the radial operator angle-dependent, but it
    remains uncoupled between theta columns.  Exact mirror columns form the
    initial groups.  Neighboring mirror orbits (or neighboring columns for an
    asymmetric metric) may then share a representative factor only when their
    complete radial coefficient profiles differ by at most 10 percent.  This
    deliberately approximate inverse is used only as a Krylov preconditioner;
    it does not alter the exact coupled stage operator or convergence check.

    Three representative factor sets are cached for the LS-IMEX-RK3 diagonal
    coefficients and rebuilt only when ``dt`` changes.
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
        self._factor_groups = self._build_factor_groups()
        self._identity = np.eye(grid.nr - 2)
        self._dt: float | None = None
        self._factors: list[list[tuple[np.ndarray, np.ndarray]]] = []
        self._wall_responses: list[np.ndarray] = []
        self.factorizations = 0
        self.factor_seconds = 0.0
        self.solve_seconds = 0.0

    @staticmethod
    def _profiles_match(
        first: np.ndarray,
        second: np.ndarray,
        tolerance: float,
    ) -> bool:
        """Return whether complete radial profiles meet a relative bound."""
        scale = np.maximum(
            np.maximum(np.abs(first), np.abs(second)),
            np.finfo(float).tiny,
        )
        return bool(np.all(np.abs(first - second) <= tolerance * scale))

    def _build_exact_mirror_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return roundoff-equivalent mirror pairs or singleton columns.

        The symmetric Joukowski map produces coefficient columns ``j`` and
        ``-j`` that differ only by floating-point evaluation order.  Use a
        tight relative test over the complete radial profile before sharing a
        factor, so cambered/asymmetric geometries preserve their distinct
        columns.  The scan order is also the natural mirror-orbit order for a
        symmetric map: ``0, (1, -1), (2, -2), ...``.
        """
        groups: list[tuple[int, ...]] = []
        claimed = np.zeros(self.grid.ntheta, dtype=bool)
        for column in range(self.grid.ntheta):
            if claimed[column]:
                continue
            mirror = (-column) % self.grid.ntheta
            if mirror != column and not claimed[mirror]:
                first = self._coefficient[:, column]
                second = self._coefficient[:, mirror]
                if self._profiles_match(
                    first,
                    second,
                    _EXACT_MIRROR_TOLERANCE,
                ):
                    groups.append((column, mirror))
                    claimed[column] = True
                    claimed[mirror] = True
                    continue
            groups.append((column,))
            claimed[column] = True
        return tuple(groups)

    def _build_factor_groups(self) -> tuple[tuple[int, ...], ...]:
        """Greedily merge adjacent exact groups within the profile bound.

        Starting from exact mirror groups preserves symmetry without assuming
        it.  On a symmetric map these are consecutive mirror orbits; on an
        asymmetric map they naturally fall back to consecutive single theta
        columns.  Every member is checked against the factored representative
        over all interior radial nodes before a group is extended.
        """
        exact_groups = self._build_exact_mirror_groups()
        merged: list[tuple[int, ...]] = []
        current: list[int] = []
        representative = -1
        for exact_group in exact_groups:
            if not current:
                current = list(exact_group)
                representative = exact_group[0]
                continue
            reference_profile = self._coefficient[:, representative]
            if all(
                self._profiles_match(
                    reference_profile,
                    self._coefficient[:, column],
                    RADIAL_PRECONDITIONER_PROFILE_TOLERANCE,
                )
                for column in exact_group
            ):
                current.extend(exact_group)
            else:
                merged.append(tuple(current))
                current = list(exact_group)
                representative = exact_group[0]
        if current:
            merged.append(tuple(current))
        return tuple(merged)

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
        # Drop the old O(Ngroup*Nr^2) cache before constructing its replacement
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
            for group in self._factor_groups:
                column = group[0]
                matrix = self._identity - scale * (
                    self._coefficient[:, column, None] * self._dii
                )
                factor = linalg.lu_factor(matrix, check_finite=False)
                stage_factors.append(factor)
                response[:, group] = linalg.lu_solve(
                    factor,
                    response_rhs[:, group],
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
        """Return the preconditioner's response to unit wall vorticity.

        For each representative matrix this is

        ``(I - alpha*dt*L_II)^-1 alpha*dt*L_I0``.

        Group members use that representative inverse with their own exact
        boundary forcing.  The response remains diagonal in physical theta,
        so multiplying it by a wall vector broadcasts the complete interior
        preconditioner influence field.
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
        """Apply the grouped radial preconditioner with supplied boundaries."""
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
        interior = solve_cached_groups(
            self._factors[stage],
            self._factor_groups,
            interior_rhs,
        )
        out = np.empty_like(rhs)
        out[0] = wall
        out[1:-1] = interior
        out[-1] = outer
        self.solve_seconds += time.perf_counter() - started
        return np.ascontiguousarray(out)

    def residual(self, field: np.ndarray, rhs: np.ndarray, stage: int) -> float:
        """Measure defect against the ungrouped radial operator.

        This is generally nonzero for nonrepresentative columns because this
        class intentionally applies an approximate preconditioner inverse.
        """
        alpha_dt = LS_IMEX_ALPHA[stage] * float(self._dt or 0.0)
        defect = field - alpha_dt * self.apply(field) - rhs
        numerator = float(np.max(np.abs(defect[1:-1])))
        denominator = max(1.0, float(np.max(np.abs(rhs[1:-1]))))
        return numerator / denominator


class ThomWallInfluence:
    """Cached Schur complements for the local Thom wall closure.

    The grouped radial preconditioner is diagonal in physical theta: its
    response to a wall-vorticity vector is
    ``R(s, theta) * omega_wall(theta)``.  The Poisson solve couples those
    columns in theta.  If ``B`` is the local cubic Thom operator, the
    preconditioner closure satisfies

    ``omega_wall = B(psi_base) + B(Poisson(R * omega_wall))``.

    This class assembles and factors the dense ``I - B P R`` preconditioner
    system once per ``(dt, stage)``.  Unlike a direct wall-derivative
    projection, this system tends continuously to the identity as ``dt``
    tends to zero.
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

        near_wall = self.poisson.solve_modes_homogeneous_near_wall(source)
        modal_curvature = -(
            self._wall_weights[0] * near_wall[:, 0]
            + self._wall_weights[1] * near_wall[:, 1]
        )

        mode_count = modal_curvature.shape[0]
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
