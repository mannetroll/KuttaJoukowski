"""Unsteady vorticity-streamfunction Navier-Stokes solver."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gcrotmk

from .mapping import AirfoilMapping
from .spectral import SpectralGrid
from .poisson import PoissonSolver
from .boundary import (
    update_wall_vorticity,
    wall_vorticity_from_streamfunction,
)
from .diagnostics import circulation_from_streamfunction, kinetic_energy
from .imex import (
    LS_IMEX_ALPHA,
    LS_IMEX_BETA,
    MappedImplicitDiffusion,
    RadialImplicitDiffusion,
    ThomWallInfluence,
)


@dataclass
class SolverConfig:
    re: float = 10000.0
    alpha: float = 5.0
    u_inf: float = 1.0
    nr: int = 240
    ntheta: int = 768
    cfl: float = 0.4
    outer_radius: float = 15.0
    thickness: float = 0.12
    camber: float = 0.0
    initial_layer_scale: float = 0.08

    @property
    def nu(self) -> float:
        return self.u_inf / self.re


class FlowSolver:
    """LS-IMEX-RK3 discretization on a conformal Fourier-Chebyshev grid."""

    def __init__(self, config: SolverConfig | None = None):
        self.config = config or SolverConfig()
        if (self.config.re <= 0 or self.config.u_inf <= 0 or
                self.config.cfl <= 0 or self.config.initial_layer_scale <= 0):
            raise ValueError("Re, U_inf, CFL and initial layer scale must be positive")
        self.mapping = AirfoilMapping(self.config.thickness, self.config.camber,
                                      self.config.outer_radius)
        self.grid = SpectralGrid(self.config.nr, self.config.ntheta, self.mapping.s_max)
        S, T = np.meshgrid(self.grid.s, self.grid.theta, indexing="ij")
        self.S, self.T = S, T
        self.x, self.y = self.mapping.computational_to_physical(S, T)
        self.H = self.mapping.metric(S, T)
        self.H2 = np.ascontiguousarray(self.H * self.H)
        if not np.all(np.isfinite(self.H)) or self.H.min() <= 0:
            raise FloatingPointError("mapping metric is not finite and positive")
        self.es, self.et = self.mapping.coordinate_basis(S, T)
        self.poisson = PoissonSolver(self.grid)
        self.omega = np.zeros((self.config.nr, self.config.ntheta), dtype=float)
        self.time = 0.0; self.step_count = 0; self.last_dt = 0.0
        self.last_cfl = 0.0; self.pressure: np.ndarray | None = None
        self.last_dt_adv = 0.0; self.last_dt_theta = 0.0
        self.last_dt_diff_explicit = 0.0; self.last_implicit_residual = 0.0
        self.last_wall_iterations = 0
        self.last_implicit_operator_applications = 0
        self._velocity_cache: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ] | None = None
        self.coarse_timestep_fallback = self.config.nr < 40
        self.timing = {
            "poisson": 0.0,
            "rhs": 0.0,
            "nonlinear": 0.0,
            "implicit": 0.0,
            "pressure": 0.0,
        }
        alpha = np.deg2rad(self.config.alpha)
        self.outer_psi_target = self.config.u_inf * (np.cos(alpha) * self.y[-1] - np.sin(alpha) * self.x[-1])
        self.wall_psi_target = float(np.mean(self.outer_psi_target))
        q = np.clip((self.S / self.mapping.s_max - 0.85) / 0.15, 0.0, 1.0)
        self.sponge_strength = 0.25 * q * q
        self.radial_implicit = RadialImplicitDiffusion(
            self.grid,
            self.H2,
            self.config.nu,
        )
        self.implicit_diffusion = MappedImplicitDiffusion(
            self.grid,
            self.H2,
            self.config.nu,
        )
        self.wall_influence = ThomWallInfluence(
            self.grid,
            self.H2,
            self.poisson,
            self.radial_implicit,
        )
        self._initialize_compatible_flow()

    def _initialize_compatible_flow(self) -> None:
        """Construct full-strength, discretely compatible no-slip initial data.

        Start from the harmonic slip flow, then subtract a smooth radial lift
        of its wall-tangential derivative.  The lift vanishes at both radial
        boundaries and has unit discrete derivative at the wall, so the
        resulting streamfunction satisfies no penetration, no slip, and the
        full outer free stream without a timestep-count startup ramp.
        """
        zero_rhs = np.zeros_like(self.H2)
        psi_slip = self.poisson.solve(
            zero_rhs,
            wall=self.wall_psi_target,
            outer=self.outer_psi_target,
        )
        wall_derivative = self.grid.Ds[0] @ psi_slip
        exponent = min(
            max(2, round(self.mapping.s_max / self.config.initial_layer_scale)),
            max(2, (self.config.nr - 1) // 3),
        )
        radial_lift = self.grid.s * (
            1.0 - self.grid.s / self.mapping.s_max
        ) ** exponent
        lift_derivative = float(self.grid.Ds[0] @ radial_lift)
        if not np.isfinite(lift_derivative) or abs(lift_derivative) < 1e-14:
            raise FloatingPointError("invalid compatible-initialization lift")
        radial_lift /= lift_derivative

        psi_target = psi_slip - radial_lift[:, None] * wall_derivative[None, :]
        omega = -self.grid.laplacian_coordinate(psi_target) / self.H2
        omega[-1, :] = 0.0
        update_wall_vorticity(
            omega,
            psi_target,
            self.grid.Dss,
            self.H2,
            self.grid.s,
        )
        psi = self.poisson.solve(
            -self.H2 * omega,
            wall=self.wall_psi_target,
            outer=self.outer_psi_target,
        )
        compatibility_error = float(np.max(np.abs(psi - psi_target)))
        if (not np.all(np.isfinite(omega)) or not np.all(np.isfinite(psi)) or
                compatibility_error > 1e-7):
            raise FloatingPointError(
                "compatible initialization failed "
                f"(streamfunction residual {compatibility_error:.3e})"
            )
        self.omega = np.ascontiguousarray(omega)
        # Keep the algebraically constructed compatible target.  Re-solving
        # Poisson changes it by only roundoff, but the tightly clustered cubic
        # Thom weights amplify that roundoff into a visible wall mismatch.
        self.psi = np.ascontiguousarray(psi_target)
        self._velocity_cache = None
        self.initial_compatibility_error = compatibility_error

    def solve_streamfunction(self, omega: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.poisson.solve(
            -self.H2 * omega,
            wall=self.wall_psi_target,
            outer=self.outer_psi_target,
        )
        self.timing["poisson"] += time.perf_counter() - t0
        return out

    def _solve_streamfunction_response(self, omega: np.ndarray) -> np.ndarray:
        """Solve a vorticity response with homogeneous streamfunction data."""
        t0 = time.perf_counter()
        out = self.poisson.solve(
            -self.H2 * omega,
            wall=0.0,
            outer=0.0,
        )
        self.timing["poisson"] += time.perf_counter() - t0
        return out

    def velocity(self, psi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if psi is None and self._velocity_cache is not None:
            return self._velocity_cache
        p = self.psi if psi is None else psi
        ps = self.grid.s_derivative(p)
        pt = self.grid.theta_derivative(p)
        qs = pt / self.H
        qt = -ps / self.H
        # The conformal coordinate basis is geometry-only and was already
        # precomputed during initialization.  Reconstructing it here created
        # complex mapping arrays on every CFL/diagnostic evaluation.
        physical_velocity = qs * self.es + qt * self.et
        u, v = physical_velocity.real, physical_velocity.imag
        result = (
            np.ascontiguousarray(u),
            np.ascontiguousarray(v),
            np.ascontiguousarray(qs),
            np.ascontiguousarray(qt),
        )
        if psi is None:
            self._velocity_cache = result
        return result

    def nonlinear_term(self, omega: np.ndarray, psi: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        ps = self.grid.s_derivative(psi); pt = self.grid.theta_derivative(psi)
        ws = self.grid.s_derivative(omega); wt = self.grid.theta_derivative(omega)
        out = self.grid.dealias_nonlinear((ps * wt - pt * ws) / self.H2)
        self.timing["nonlinear"] += time.perf_counter() - t0
        return out

    def rhs(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the complete semidiscrete RHS for diagnostics and tests."""
        t0 = time.perf_counter()
        psi = self.solve_streamfunction(omega)
        out = self._explicit_rhs(omega, psi)
        out += self.implicit_diffusion.apply(omega)
        out[[0, -1], :] = 0.0
        self.timing["rhs"] += time.perf_counter() - t0
        return out, psi

    def _explicit_rhs(self, omega: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Advection, sponge damping, and other nonstiff explicit terms."""
        adv = self.nonlinear_term(omega, psi)
        out = -adv - self.sponge_strength * omega
        out[[0, -1], :] = 0.0
        return np.ascontiguousarray(out)

    def timestep_limits(self) -> tuple[float, float, float]:
        """Return advective, explicit-theta, and all-explicit diffusion limits."""
        _, _, qs, qt = self.velocity()
        ds = np.empty(self.config.nr)
        gaps = np.diff(self.grid.s)
        ds[0] = gaps[0]; ds[-1] = gaps[-1]
        ds[1:-1] = np.minimum(gaps[:-1], gaps[1:])
        dtheta = 2 * np.pi / self.config.ntheta
        speed_s = np.abs(qs / self.H)
        speed_t = np.abs(qt / self.H)
        rate = speed_s / ds[:, None] + speed_t / dtheta
        dt_adv = self.config.cfl / max(float(np.max(rate[1:-1])), 1e-12)
        inv_theta = 1.0 / (self.H2 * dtheta**2)
        dt_theta = 0.22 / max(
            self.config.nu * float(np.max(inv_theta[1:-1])),
            1e-12,
        )
        inv_explicit = 1.0 / (self.H2 * ds[:, None] ** 2) + inv_theta
        dt_diff_explicit = 0.22 / max(
            self.config.nu * float(np.max(inv_explicit[1:-1])),
            1e-12,
        )
        return float(dt_adv), float(dt_theta), float(dt_diff_explicit)

    def stable_timestep(self) -> float:
        dt_adv, dt_theta, dt_diff_explicit = self.timestep_limits()
        # These are the limits that select the upcoming step and therefore
        # the appropriate values for its reported CFL.  Recomputing them
        # after every completed step duplicated a full velocity recovery.
        self.last_dt_adv = dt_adv
        self.last_dt_theta = dt_theta
        self.last_dt_diff_explicit = dt_diff_explicit
        # Both coordinate directions of viscosity are implicit.  Keep the old
        # explicit limits as diagnostics, but do not let either select a
        # resolved-grid production step.
        candidate = min(dt_adv, 0.02)
        if self.coarse_timestep_fallback:
            # Small grids are useful for smoke tests but do not resolve
            # the initial wall layer or the cubic Thom reconstruction.  Keep
            # their legacy explicit radial bound and a small absolute cap;
            # resolved grids use the substantially larger IMEX limit above.
            candidate = min(candidate, dt_diff_explicit, 2e-5)
        dt = candidate
        if self.last_dt > 0.0 and not self.coarse_timestep_fallback:
            # Rebuilding every radial and wall factor for insignificant
            # advective-CFL changes is more expensive than advancing a step.
            # Reuse a safe accepted dt, increase only with 25% hysteresis, and
            # leave 10% headroom whenever a smaller advective dt is required.
            if self.last_dt <= candidate < 1.25 * self.last_dt:
                dt = self.last_dt
            elif candidate < self.last_dt:
                dt = 0.9 * candidate
            elif candidate >= 1.25 * self.last_dt:
                # The candidate itself already satisfies the requested CFL.
                # Once a factor rebuild is justified, take the whole safe gain
                # rather than paying that cost for only 90% of it.
                dt = candidate
        if not np.isfinite(dt) or dt <= 0:
            raise FloatingPointError("invalid adaptive timestep")
        return float(dt)

    def _apply_boundary(self, omega: np.ndarray, psi: np.ndarray) -> None:
        update_wall_vorticity(omega, psi, self.grid.Dss, self.H2, self.grid.s)

    def _wall_response_from_vorticity(self, omega: np.ndarray) -> np.ndarray:
        """Apply the homogeneous Poisson-plus-Thom wall functional."""
        near_wall = self.poisson.solve_homogeneous_near_wall(
            -self.H2 * omega
        )
        weights = self.wall_influence._wall_weights
        curvature = weights[0] * near_wall[0] + weights[1] * near_wall[1]
        return np.ascontiguousarray(-curvature / self.H2[0])

    def _radial_closed_solve(
        self,
        rhs: np.ndarray,
        stage: int,
    ) -> np.ndarray:
        """Apply the cached radial/Thom inverse used as a preconditioner.

        Unlike the old stage routine, ``rhs[0]`` is an arbitrary right-hand
        side for the *linear* wall equation.  This makes the action suitable
        as a Krylov preconditioner for the coupled two-dimensional operator.
        """
        zero_wall = np.zeros(self.config.ntheta, dtype=float)
        omega_base = self.radial_implicit.solve(
            rhs,
            stage,
            zero_wall,
            outer=rhs[-1],
        )
        closure_rhs = rhs[0] + self._wall_response_from_vorticity(omega_base)
        wall = self.wall_influence.solve(stage, closure_rhs)

        omega = omega_base
        omega[0] = wall
        omega[1:-1] += (
            self.radial_implicit.wall_response(stage) * wall[None, :]
        )
        return np.ascontiguousarray(omega)

    def _implicit_operator_apply(
        self,
        omega: np.ndarray,
        alpha_dt: float,
    ) -> np.ndarray:
        """Apply the exact coupled stage matrix, including wall equations."""
        out = omega - alpha_dt * self.implicit_diffusion.apply(omega)
        out[0] = omega[0] - self._wall_response_from_vorticity(omega)
        out[-1] = omega[-1]
        return np.ascontiguousarray(out)

    def _implicit_stage(
        self,
        rhs: np.ndarray,
        stage: int,
        reference_omega: np.ndarray,
        reference_psi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, int, int]:
        """Solve one full mapped-diffusion stage with preconditioned GCROT.

        The Krylov unknown is the increment from the current compatible state.
        Its wall equation is homogeneous, avoiding cancellation between a
        million-scale affine Thom forcing and homogeneous Poisson response.
        """
        alpha_dt = LS_IMEX_ALPHA[stage] * float(self.radial_implicit._dt or 0.0)
        shape = rhs.shape
        size = rhs.size
        linear_rhs = rhs - (
            reference_omega
            - alpha_dt * self.implicit_diffusion.apply(reference_omega)
        )
        linear_rhs[0] = (
            wall_vorticity_from_streamfunction(
                reference_psi,
                self.H2,
                self.grid.s,
            )
            - reference_omega[0]
        )
        linear_rhs[-1] = -reference_omega[-1]
        flat_rhs = linear_rhs.ravel()

        operator_applications = 0

        def apply_operator(values: np.ndarray) -> np.ndarray:
            nonlocal operator_applications
            operator_applications += 1
            return self._implicit_operator_apply(
                values.reshape(shape),
                alpha_dt,
            ).ravel()

        operator = LinearOperator(
            (size, size),
            matvec=apply_operator,
            dtype=float,
        )
        preconditioner = LinearOperator(
            (size, size),
            matvec=lambda values: self._radial_closed_solve(
                values.reshape(shape),
                stage,
            ).ravel(),
            dtype=float,
        )
        # Applying the full radial/Thom preconditioner only to seed ``x0``
        # costs one grouped-LU/Poisson solve per stage.  The unpreconditioned
        # linear RHS is already a close increment for these small IMEX stage
        # corrections; GCROT still applies the same preconditioner to its
        # Krylov vectors and the full residual is checked independently.
        initial = linear_rhs.ravel().copy()
        solution, info = gcrotmk(
            operator,
            flat_rhs,
            x0=initial,
            M=preconditioner,
            m=10,
            k=5,
            maxiter=30,
            rtol=3e-7,
            atol=0.0,
        )
        if info != 0:
            raise FloatingPointError(
                "full implicit diffusion solve did not converge "
                f"at stage {stage} (GCROT info={info}, "
                f"operator applications={operator_applications})"
            )

        def recover_and_measure(values: np.ndarray):
            correction = np.ascontiguousarray(values.reshape(shape))
            omega = np.ascontiguousarray(reference_omega + correction)
            psi = np.ascontiguousarray(
                reference_psi + self._solve_streamfunction_response(correction)
            )
            interior_defect = (
                omega
                - alpha_dt * self.implicit_diffusion.apply(omega)
                - rhs
            )
            wall_target = wall_vorticity_from_streamfunction(
                psi,
                self.H2,
                self.grid.s,
            )
            numerator = max(
                float(np.max(np.abs(interior_defect[1:-1]))),
                float(np.max(np.abs(omega[0] - wall_target))),
                float(np.max(np.abs(omega[-1]))),
            )
            denominator = max(1.0, float(np.max(np.abs(rhs[1:-1]))))
            return omega, psi, numerator / denominator

        omega, psi, residual = recover_and_measure(solution)
        if not np.isfinite(residual) or residual > 1e-5:
            # GCROT's normwise convergence test can be looser than the
            # independently checked max residual on very stiff low-Re cases.
            # Refine only those stages, starting from the already good result;
            # normal Re=1000/10000 production steps avoid this extra work.
            solution, info = gcrotmk(
                operator,
                flat_rhs,
                x0=solution,
                M=preconditioner,
                m=10,
                k=5,
                maxiter=30,
                rtol=1e-10,
                atol=0.0,
            )
            if info != 0:
                raise FloatingPointError(
                    "full implicit diffusion refinement did not converge "
                    f"at stage {stage} (GCROT info={info}, "
                    f"operator applications={operator_applications})"
                )
            omega, psi, residual = recover_and_measure(solution)
        if not np.isfinite(residual) or residual > 1e-5:
            raise FloatingPointError(
                "full implicit diffusion residual exceeded tolerance "
                f"at stage {stage} ({residual:.3e})"
            )
        return omega, np.ascontiguousarray(psi), residual, 1, operator_applications

    def step(
        self,
        dt: float | None = None,
        *,
        return_diagnostics: bool = True,
    ) -> dict[str, float]:
        if dt is None:
            dt = self.stable_timestep()
            limits = (
                self.last_dt_adv,
                self.last_dt_theta,
                self.last_dt_diff_explicit,
            )
        else:
            dt = float(dt)
            limits = self.timestep_limits()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        implicit_elapsed = 0.0
        prepare_started = time.perf_counter()
        self.radial_implicit.prepare(dt)
        self.wall_influence.prepare(dt)
        implicit_elapsed += time.perf_counter() - prepare_started
        omega = self.omega.copy()
        psi = self.psi.copy()
        previous_explicit = np.zeros_like(omega)
        maximum_residual = 0.0
        maximum_wall_iterations = 0
        maximum_operator_applications = 0
        for stage, (alpha, beta) in enumerate(zip(LS_IMEX_ALPHA, LS_IMEX_BETA)):
            explicit = self._explicit_rhs(omega, psi)
            implicit_started = time.perf_counter()
            stage_rhs = (
                omega
                + beta * dt * self.implicit_diffusion.apply(omega)
                + alpha * dt * explicit
                + beta * dt * previous_explicit
            )
            omega, psi, residual, wall_iterations, operator_applications = self._implicit_stage(
                stage_rhs,
                stage,
                omega,
                psi,
            )
            implicit_elapsed += time.perf_counter() - implicit_started
            previous_explicit = explicit
            maximum_residual = max(maximum_residual, residual)
            maximum_wall_iterations = max(maximum_wall_iterations, wall_iterations)
            maximum_operator_applications = max(
                maximum_operator_applications,
                operator_applications,
            )
        omega[-1] = 0.0
        if not np.all(np.isfinite(omega)) or np.max(np.abs(omega)) > 1e10:
            raise FloatingPointError("vorticity solution became unstable")
        self.omega = np.ascontiguousarray(omega)
        self.psi = np.ascontiguousarray(psi)
        self._velocity_cache = None
        self.time += dt; self.step_count += 1; self.last_dt = dt
        self.last_implicit_residual = maximum_residual
        self.last_wall_iterations = maximum_wall_iterations
        self.last_implicit_operator_applications = maximum_operator_applications
        self.timing["implicit"] += implicit_elapsed
        dt_adv, dt_theta, dt_diff_explicit = limits
        self.last_dt_adv = dt_adv
        self.last_dt_theta = dt_theta
        self.last_dt_diff_explicit = dt_diff_explicit
        self.last_cfl = dt / max(dt_adv, 1e-30) * self.config.cfl
        return self.diagnostics() if return_diagnostics else {}

    def compute_pressure(self) -> np.ndarray:
        from .pressure import recover_pressure
        t0 = time.perf_counter(); self.pressure = recover_pressure(self)
        self.timing["pressure"] += time.perf_counter() - t0
        return self.pressure

    def diagnostics(self) -> dict[str, float]:
        u, v, qs, qt = self.velocity()
        # velocity() already formed qt = -psi_s/H; reuse it instead of a
        # second dense Chebyshev derivative solely for circulation.
        ps = -qt * self.H
        circulation_index = min(3, self.config.nr - 2)
        gamma = circulation_from_streamfunction(ps, circulation_index)
        data = {
            "time": self.time, "step": float(self.step_count), "dt": self.last_dt,
            "cfl": self.last_cfl, "max_omega": float(np.max(np.abs(self.omega))),
            "max_velocity": float(np.max(np.hypot(u, v))), "gamma": gamma,
            "gamma_contour_s": float(self.grid.s[circulation_index]),
            "cl_circulation": 2 * gamma / self.config.u_inf,
            "wall_slip": float(np.max(np.abs(qt[0]))),
            "kinetic_energy": kinetic_energy(qs, qt, self.H2, self.grid.s),
            "initial_compatibility_error": self.initial_compatibility_error,
            "dt_adv": self.last_dt_adv,
            "dt_theta": self.last_dt_theta,
            "dt_diff_explicit": self.last_dt_diff_explicit,
            "implicit_residual": self.last_implicit_residual,
            "wall_iterations": float(self.last_wall_iterations),
            "implicit_operator_applications": float(
                self.last_implicit_operator_applications
            ),
            "coarse_timestep_fallback": float(self.coarse_timestep_fallback),
        }
        if self.pressure is not None:
            from .pressure import surface_coefficients
            coeff = surface_coefficients(self, self.pressure)
            data.update({k: float(v) for k, v in coeff.items() if np.isscalar(v)})
        return data

    def field(self, name: str) -> np.ndarray:
        key = name.lower()
        if key == "vorticity": return self.omega
        if key == "streamfunction": return self.psi
        u, v, _, _ = self.velocity()
        if key == "velocity magnitude": return np.hypot(u, v)
        if key == "u velocity": return u
        if key == "v velocity": return v
        if key in ("pressure", "cp-like pressure field"):
            p = self.pressure if self.pressure is not None else self.compute_pressure()
            return p if key == "pressure" else p / (0.5 * self.config.u_inf**2)
        raise KeyError(name)

    def config_dict(self) -> dict:
        return asdict(self.config)
