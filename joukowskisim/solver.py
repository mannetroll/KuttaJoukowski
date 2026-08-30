"""Unsteady vorticity-streamfunction Navier-Stokes solver."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import numpy as np

from .mapping import AirfoilMapping
from .spectral import SpectralGrid
from .poisson import PoissonSolver
from .boundary import update_wall_vorticity, wall_velocity_error
from .diagnostics import circulation_from_streamfunction, kinetic_energy
from .imex import (
    LS_IMEX_ALPHA,
    LS_IMEX_BETA,
    RadialImplicitDiffusion,
    ThomWallInfluence,
)


@dataclass
class SolverConfig:
    re: float = 1000.0
    alpha: float = 5.0
    u_inf: float = 1.0
    nr: int = 160
    ntheta: int = 512
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
        self.psi = np.ascontiguousarray(psi)
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
        p = self.psi if psi is None else psi
        ps = self.grid.s_derivative(p)
        pt = self.grid.theta_derivative(p)
        qs = pt / self.H
        qt = -ps / self.H
        u, v = self.mapping.velocity_computational_to_physical(qs, qt, self.S, self.T)
        return np.ascontiguousarray(u), np.ascontiguousarray(v), qs, qt

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
        out += self.radial_implicit.apply(omega)
        out[[0, -1], :] = 0.0
        self.timing["rhs"] += time.perf_counter() - t0
        return out, psi

    def _explicit_rhs(self, omega: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Terms advanced explicitly by LS-IMEX-RK3."""
        adv = self.nonlinear_term(omega, psi)
        theta_diffusion = (
            self.config.nu
            * self.grid.theta_derivative(omega, 2)
            / self.H2
        )
        out = -adv + theta_diffusion - self.sponge_strength * omega
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
        candidate = min(dt_adv, dt_theta, 0.02)
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
            elif candidate < self.last_dt and dt_adv <= dt_theta:
                dt = 0.9 * candidate
            elif candidate >= 1.25 * self.last_dt:
                dt = 0.9 * candidate if dt_adv <= dt_theta else candidate
        if not np.isfinite(dt) or dt <= 0:
            raise FloatingPointError("invalid adaptive timestep")
        return float(dt)

    def _apply_boundary(self, omega: np.ndarray, psi: np.ndarray) -> None:
        update_wall_vorticity(omega, psi, self.grid.Dss, self.H2, self.grid.s)

    def _implicit_stage(
        self,
        rhs: np.ndarray,
        stage: int,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Solve one radial stage and its dense Thom wall Schur system."""
        zero_wall = np.zeros(self.config.ntheta, dtype=float)
        omega_base = self.radial_implicit.solve(
            rhs,
            stage,
            zero_wall,
        )
        psi_base = self.solve_streamfunction(omega_base)
        closed_base = omega_base.copy()
        self._apply_boundary(closed_base, psi_base)
        closure_rhs = closed_base[0]
        wall = self.wall_influence.solve(stage, closure_rhs)

        omega = omega_base
        omega[0] = wall
        omega_response = np.zeros_like(omega)
        omega_response[1:-1] = (
            self.radial_implicit.wall_response(stage) * wall[None, :]
        )
        omega[1:-1] += omega_response[1:-1]
        # Preserve the Schur decomposition numerically.  Re-solving the full
        # nonhomogeneous problem loses ~1e-11 in psi; the tightly clustered
        # wall reconstruction can amplify that into a visible closure error.
        psi = psi_base + self._solve_streamfunction_response(omega_response)
        closed = omega.copy()
        self._apply_boundary(closed, psi)
        wall_error = float(np.max(np.abs(closed[0] - wall))) / max(
            1.0,
            float(np.max(np.abs(wall))),
        )
        schur_error = self.wall_influence.residual(
            stage,
            wall,
            closure_rhs,
        )
        residual = max(
            self.radial_implicit.residual(omega, rhs, stage),
            wall_error,
            schur_error,
        )
        return np.ascontiguousarray(omega), np.ascontiguousarray(psi), residual, 1

    def step(self, dt: float | None = None) -> dict[str, float]:
        dt = self.stable_timestep() if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        implicit_started = time.perf_counter()
        self.radial_implicit.prepare(dt)
        self.wall_influence.prepare(dt)
        omega = self.omega.copy()
        psi = self.psi.copy()
        previous_explicit = np.zeros_like(omega)
        maximum_residual = 0.0
        maximum_wall_iterations = 0
        for stage, (alpha, beta) in enumerate(zip(LS_IMEX_ALPHA, LS_IMEX_BETA)):
            explicit = self._explicit_rhs(omega, psi)
            stage_rhs = (
                omega
                + beta * dt * self.radial_implicit.apply(omega)
                + alpha * dt * explicit
                + beta * dt * previous_explicit
            )
            omega, psi, residual, wall_iterations = self._implicit_stage(
                stage_rhs,
                stage,
            )
            previous_explicit = explicit
            maximum_residual = max(maximum_residual, residual)
            maximum_wall_iterations = max(maximum_wall_iterations, wall_iterations)
        omega[-1] = 0.0
        if not np.all(np.isfinite(omega)) or np.max(np.abs(omega)) > 1e10:
            raise FloatingPointError("vorticity solution became unstable")
        self.omega = np.ascontiguousarray(omega)
        self.psi = np.ascontiguousarray(psi)
        self.time += dt; self.step_count += 1; self.last_dt = dt
        self.last_implicit_residual = maximum_residual
        self.last_wall_iterations = maximum_wall_iterations
        self.timing["implicit"] += time.perf_counter() - implicit_started
        dt_adv, dt_theta, dt_diff_explicit = self.timestep_limits()
        self.last_dt_adv = dt_adv
        self.last_dt_theta = dt_theta
        self.last_dt_diff_explicit = dt_diff_explicit
        self.last_cfl = dt / max(dt_adv, 1e-30) * self.config.cfl
        return self.diagnostics()

    def compute_pressure(self) -> np.ndarray:
        from .pressure import recover_pressure
        t0 = time.perf_counter(); self.pressure = recover_pressure(self)
        self.timing["pressure"] += time.perf_counter() - t0
        return self.pressure

    def diagnostics(self) -> dict[str, float]:
        u, v, qs, qt = self.velocity()
        ps = self.grid.s_derivative(self.psi)
        gamma = circulation_from_streamfunction(ps, min(3, self.config.nr - 2))
        data = {
            "time": self.time, "step": float(self.step_count), "dt": self.last_dt,
            "cfl": self.last_cfl, "max_omega": float(np.max(np.abs(self.omega))),
            "max_velocity": float(np.max(np.hypot(u, v))), "gamma": gamma,
            "cl_kj": 2 * gamma / self.config.u_inf,
            "wall_slip": wall_velocity_error(self.psi, self.grid.Ds, self.H),
            "kinetic_energy": kinetic_energy(qs, qt, self.H2, self.grid.s),
            "initial_compatibility_error": self.initial_compatibility_error,
            "dt_adv": self.last_dt_adv,
            "dt_theta": self.last_dt_theta,
            "dt_diff_explicit": self.last_dt_diff_explicit,
            "implicit_residual": self.last_implicit_residual,
            "wall_iterations": float(self.last_wall_iterations),
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
