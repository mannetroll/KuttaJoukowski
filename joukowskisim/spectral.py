"""Fourier/Chebyshev spectral operators (SciPy FFT, all CPU workers)."""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy import fft


def chebyshev_lobatto(n: int, s_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 4:
        raise ValueError("Nr must be at least 4")
    j = np.arange(n)
    x = np.cos(np.pi * j / (n - 1))
    c = np.ones(n); c[[0, -1]] = 2
    c *= (-1.0) ** j
    X = np.tile(x, (n, 1)).T
    dX = X - X.T
    D = (np.outer(c, 1.0 / c) / (dX + np.eye(n)))
    D -= np.diag(D.sum(axis=1))
    # x decreases as s increases.
    s = 0.5 * s_max * (1.0 - x)
    Ds = (-2.0 / s_max) * D
    Dss = Ds @ Ds
    return np.ascontiguousarray(s), np.ascontiguousarray(Ds), np.ascontiguousarray(Dss)


@dataclass
class SpectralGrid:
    nr: int
    ntheta: int
    s_max: float

    def __post_init__(self) -> None:
        if self.ntheta < 8 or self.ntheta % 2:
            raise ValueError("Ntheta must be even and at least 8")
        self.s, self.Ds, self.Dss = chebyshev_lobatto(self.nr, self.s_max)
        self.theta = 2 * np.pi * np.arange(self.ntheta) / self.ntheta
        self.k = fft.fftfreq(self.ntheta, 1.0 / self.ntheta)
        self.kr = fft.rfftfreq(self.ntheta, 1.0 / self.ntheta)
        cutoff = self.ntheta // 3
        self.dealias_mask = np.abs(self.k) <= cutoff
        radial_cutoff = 2 * (self.nr - 1) // 3
        radial_mode = np.arange(self.nr)
        radial_eta = np.clip(
            (radial_mode - radial_cutoff)
            / max(1, self.nr - 1 - radial_cutoff),
            0.0,
            1.0,
        )
        # Preserve the resolved two-thirds exactly, then smoothly suppress the
        # unresolved Chebyshev tail.  This is applied only to nonlinear
        # products; filtering the state itself damages the no-slip wall layer.
        self.radial_nonlinear_filter = np.exp(
            np.log(np.finfo(float).eps) * radial_eta**8
        )
        self.radial_nonlinear_filter[:radial_cutoff + 1] = 1.0
        self.fft_time = 0.0

    def theta_derivative(self, f: np.ndarray, order: int = 1) -> np.ndarray:
        t0 = time.perf_counter()
        fh = fft.fft(f, axis=-1, workers=-1)
        out = fft.ifft(fh * (1j * self.k) ** order, axis=-1, workers=-1)
        self.fft_time += time.perf_counter() - t0
        return np.ascontiguousarray(out.real)

    def s_derivative(self, f: np.ndarray, order: int = 1) -> np.ndarray:
        D = self.Ds if order == 1 else self.Dss
        return np.ascontiguousarray(D @ f)

    def dealias(self, f: np.ndarray) -> np.ndarray:
        """Remove unresolved Fourier modes along the periodic direction."""
        t0 = time.perf_counter()
        fh = fft.fft(f, axis=-1, workers=-1)
        fh[..., ~self.dealias_mask] = 0
        out = np.ascontiguousarray(fft.ifft(fh, axis=-1, workers=-1).real)
        self.fft_time += time.perf_counter() - t0
        return out

    def dealias_nonlinear(self, f: np.ndarray) -> np.ndarray:
        """Cut Fourier modes and damp the radial tail of a nonlinear product.

        The DCT-I is the Chebyshev transform on the Lobatto nodes.  Its smooth
        exponential tail filter suppresses unresolved radial content and the
        ringing it can excite, but it cannot undo aliases already folded into
        retained modes when the product was formed.  True radial dealiasing
        requires padded/overintegrated product evaluation.  Boundary
        conditions are imposed by the caller after this nonlinear-only filter.
        """
        values = np.asarray(f, dtype=float)
        if values.shape != (self.nr, self.ntheta):
            raise ValueError(
                "nonlinear dealiasing expects an (Nr, Ntheta) field"
            )
        angular = self.dealias(values)
        t0 = time.perf_counter()
        coefficients = fft.dct(angular, type=1, axis=0, workers=-1)
        coefficients *= self.radial_nonlinear_filter[:, None]
        out = fft.idct(coefficients, type=1, axis=0, workers=-1)
        self.fft_time += time.perf_counter() - t0
        return np.ascontiguousarray(out)

    def laplacian_coordinate(self, f: np.ndarray) -> np.ndarray:
        return self.Dss @ f + self.theta_derivative(f, 2)
