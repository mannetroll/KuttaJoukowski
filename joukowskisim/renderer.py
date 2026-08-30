"""Precomputed physical-pixel to curvilinear-grid rasterization."""

from __future__ import annotations
import numpy as np
from .colormaps import colorize


class CurvilinearRenderer:
    def __init__(self, solver, width: int = 1000, height: int = 520,
                 bounds: tuple[float, float, float, float] = (-1.25, 4.0, -1.5, 1.5)):
        self.solver = solver; self.width = width; self.height = height; self.bounds = bounds
        xmin, xmax, ymin, ymax = bounds
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, width), np.linspace(ymax, ymin, height))
        zn = xx + 1j * yy
        raw = zn * solver.mapping.raw_chord + complex(solver.mapping.x_min, solver.mapping.y_shift)
        disc = np.sqrt(raw * raw - 4 * solver.mapping.a**2)
        z1 = 0.5 * (raw + disc); z2 = 0.5 * (raw - disc)
        c = solver.mapping.center
        zeta = np.where(np.abs(z1-c) >= np.abs(z2-c), z1, z2)
        radius = np.abs(zeta-c)
        ss = np.log(np.maximum(radius, 1e-300) / solver.mapping.circle_radius)
        tt = np.mod(np.angle(zeta-c), 2*np.pi)
        valid = (ss >= 0) & (ss <= solver.mapping.s_max)
        ir = np.searchsorted(solver.grid.s, ss, side="right") - 1
        ir = np.clip(ir, 0, solver.config.nr-2)
        wr = (ss - solver.grid.s[ir]) / (solver.grid.s[ir+1] - solver.grid.s[ir])
        jt = tt / (2*np.pi) * solver.config.ntheta
        it = np.floor(jt).astype(int) % solver.config.ntheta
        wt = jt - np.floor(jt)
        self.valid = valid; self.ir = ir; self.it = it
        self.wr = wr; self.wt = wt
        self._surface_pixels = self._make_surface_pixels()

    def interpolate(self, field: np.ndarray) -> np.ndarray:
        i, j = self.ir, self.it; jp = (j+1) % self.solver.config.ntheta
        a = field[i,j]*(1-self.wt) + field[i,jp]*self.wt
        b = field[i+1,j]*(1-self.wt) + field[i+1,jp]*self.wt
        return a*(1-self.wr) + b*self.wr

    def render(self, field: np.ndarray, positive: bool = False) -> np.ndarray:
        return colorize(self.interpolate(field), positive=positive, valid=self.valid)

    def _make_surface_pixels(self) -> np.ndarray:
        x, y = self.solver.mapping.surface_xy(800)
        xmin, xmax, ymin, ymax = self.bounds
        px = (x-xmin)/(xmax-xmin)*self.width
        py = (ymax-y)/(ymax-ymin)*self.height
        return np.column_stack((px,py))

    def surface_pixels(self) -> np.ndarray:
        return self._surface_pixels
