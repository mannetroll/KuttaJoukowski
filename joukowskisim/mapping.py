"""Analytic regularized Joukowski airfoil mapping."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AirfoilMapping:
    """Map the exterior of an offset circle to a chord-normalized airfoil.

    ``te_gap`` separates the circle's rightmost point from the Joukowski
    critical point.  Thus f' never vanishes on or outside the circle and the
    trailing edge is thin and rounded rather than a singular cusp.
    """

    thickness: float = 0.12
    camber: float = 0.0
    outer_radius: float = 15.0
    te_gap: float = 0.015
    circle_radius: float = 1.0

    def __post_init__(self) -> None:
        # A left offset creates airfoil-like thickness; vertical offset camber.
        offset = 0.045 + 0.48 * float(self.thickness)
        self.center = complex(-offset, float(self.camber) * 0.35)
        right = self.center.real + self.circle_radius
        self.a = max(0.2, right - float(self.te_gap))
        margin = 64.0 * np.finfo(float).eps * max(
            1.0,
            self.circle_radius,
            abs(self.center),
            abs(self.a),
        )
        critical_distances = (
            abs(complex(self.a) - self.center),
            abs(complex(-self.a) - self.center),
        )
        if max(critical_distances) >= self.circle_radius - margin:
            raise ValueError(
                "Joukowski critical points must lie strictly inside the "
                "generating circle; reduce camber/thickness"
            )
        th = np.linspace(0.0, 2 * np.pi, 16384, endpoint=False)
        raw = self._raw_map(self.center + self.circle_radius * np.exp(1j * th))
        self.x_min = float(raw.real.min())
        self.x_max = float(raw.real.max())
        self.raw_chord = self.x_max - self.x_min
        self.y_shift = 0.5 * float(raw.imag.max() + raw.imag.min())
        # At large radius f(zeta) ~ zeta; this makes the far radius chord-based.
        r_outer = max(self.circle_radius * 1.05, self.outer_radius * self.raw_chord)
        self.s_max = float(np.log(r_outer / self.circle_radius))

    def _raw_map(self, zeta: np.ndarray | complex) -> np.ndarray:
        zeta = np.asarray(zeta, dtype=np.complex128)
        return zeta + self.a**2 / zeta

    def zeta(self, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
        r = self.circle_radius * np.exp(np.asarray(s))
        return self.center + r * np.exp(1j * np.asarray(theta))

    def dz_dzeta(self, zeta: np.ndarray) -> np.ndarray:
        return (1.0 - self.a**2 / np.asarray(zeta, dtype=np.complex128) ** 2) / self.raw_chord

    def z(self, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raw = self._raw_map(self.zeta(s, theta))
        return (raw - complex(self.x_min, self.y_shift)) / self.raw_chord

    def metric(self, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
        zz = self.zeta(s, theta)
        r = np.abs(zz - self.center)
        return np.ascontiguousarray(r * np.abs(self.dz_dzeta(zz)), dtype=np.float64)

    def coordinate_basis(self, s: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return unit vectors in increasing s and theta as complex arrays."""
        zz = self.zeta(s, theta)
        radial = (zz - self.center) * self.dz_dzeta(zz)
        es = radial / np.maximum(np.abs(radial), np.finfo(float).tiny)
        et = 1j * es
        return es, et

    def computational_to_physical(self, s: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.z(s, theta)
        return z.real, z.imag

    def velocity_computational_to_physical(
        self, q_s: np.ndarray, q_theta: np.ndarray, s: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        es, et = self.coordinate_basis(s, theta)
        vel = np.asarray(q_s) * es + np.asarray(q_theta) * et
        return vel.real, vel.imag

    def surface_xy(self, n: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        theta = np.linspace(0.0, 2 * np.pi, n, endpoint=True)
        s = np.zeros_like(theta)
        return self.computational_to_physical(s, theta)
