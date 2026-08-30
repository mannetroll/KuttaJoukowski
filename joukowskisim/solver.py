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
    startup_time: float = 0.5

    @property
    def nu(self) -> float:
        return self.u_inf / self.re


class FlowSolver:
    """Explicit SSPRK3 discretization on a conformal Fourier-Chebyshev grid."""

    def __init__(self, config: SolverConfig | None = None):
        self.config = config or SolverConfig()
        if self.config.re <= 0 or self.config.u_inf <= 0 or self.config.cfl <= 0:
            raise ValueError("Re, U_inf and CFL must be positive")
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
        self.timing = {"poisson": 0.0, "rhs": 0.0, "nonlinear": 0.0, "pressure": 0.0}
        alpha = np.deg2rad(self.config.alpha)
        self.outer_psi_target = self.config.u_inf * (np.cos(alpha) * self.y[-1] - np.sin(alpha) * self.x[-1])
        self.wall_psi_target = float(np.mean(self.outer_psi_target))
        self.psi = self.solve_streamfunction(self.omega)
        # A few deterministic compatibility iterations seed a smooth viscous wall layer.
        for _ in range(3):
            update_wall_vorticity(self.omega, self.psi, self.grid.Dss, self.H2, self.grid.s)
            self.psi = self.solve_streamfunction(self.omega)

    def solve_streamfunction(self, omega: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        # The step-count branch prevents the explicit Chebyshev viscous limit
        # from making the startup ramp take impractically many GUI frames.
        ramp = 1.0 if self.config.startup_time <= 0 else min(1.0, max(
            self.time / self.config.startup_time, self.step_count / 200.0))
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        out = self.poisson.solve(-self.H2 * omega, wall=ramp * self.wall_psi_target,
                                 outer=ramp * self.outer_psi_target)
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
        out = self.grid.dealias((ps * wt - pt * ws) / self.H2)
        self.timing["nonlinear"] += time.perf_counter() - t0
        return out

    def rhs(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        psi = self.solve_streamfunction(omega)
        adv = self.nonlinear_term(omega, psi)
        diffusion = self.config.nu * self.grid.laplacian_coordinate(omega) / self.H2
        # Smooth sponge only in the outermost 15% of logarithmic radius.
        q = np.clip((self.S / self.mapping.s_max - 0.85) / 0.15, 0.0, 1.0)
        sponge = 0.25 * q * q * omega
        out = -adv + diffusion - sponge
        out[[0, -1], :] = 0.0
        self.timing["rhs"] += time.perf_counter() - t0
        return out, psi

    def stable_timestep(self) -> float:
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
        inv_h = 1.0 / (self.H2 * ds[:, None] ** 2) + 1.0 / (self.H2 * dtheta**2)
        dt_diff = 0.22 / max(self.config.nu * float(np.max(inv_h[1:-1])), 1e-12)
        dt = min(dt_adv, dt_diff, 0.02)
        if not np.isfinite(dt) or dt <= 0:
            raise FloatingPointError("invalid adaptive timestep")
        return float(dt)

    def _apply_boundary(self, omega: np.ndarray, psi: np.ndarray) -> None:
        update_wall_vorticity(omega, psi, self.grid.Dss, self.H2, self.grid.s)

    def step(self, dt: float | None = None) -> dict[str, float]:
        dt = self.stable_timestep() if dt is None else float(dt)
        w0 = self.omega.copy()
        k1, p1 = self.rhs(w0); w1 = w0 + dt * k1; self._apply_boundary(w1, p1)
        k2, p2 = self.rhs(w1); w2 = 0.75 * w0 + 0.25 * (w1 + dt * k2); self._apply_boundary(w2, p2)
        k3, p3 = self.rhs(w2); w3 = (w0 + 2.0 * (w2 + dt * k3)) / 3.0; self._apply_boundary(w3, p3)
        w3[-1] = 0.0
        if not np.all(np.isfinite(w3)) or np.max(np.abs(w3)) > 1e10:
            raise FloatingPointError("vorticity solution became unstable")
        self.omega = np.ascontiguousarray(w3)
        self.psi = self.solve_streamfunction(self.omega)
        self.time += dt; self.step_count += 1; self.last_dt = dt
        self.last_cfl = dt / max(self.stable_timestep(), 1e-30) * self.config.cfl
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
