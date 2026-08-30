"""Small dependency-free RGB lookup tables."""

from __future__ import annotations
import numpy as np


def _table(stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    x = np.linspace(0, 1, 256)
    xp = np.array([p for p, _ in stops])
    colors = np.array([c for _, c in stops], float)
    return np.column_stack([np.interp(x, xp, colors[:, j]) for j in range(3)]).astype(np.uint8)


INFERNO = _table([(0,(0,0,4)),(.25,(87,16,110)),(.5,(188,55,84)),(.75,(249,142,9)),(1,(252,255,164))])
DIVERGING = _table([(0,(28,48,110)),(.25,(55,126,184)),(.5,(238,238,238)),(.75,(202,82,70)),(1,(103,0,31))])


def colorize(values: np.ndarray, positive: bool = False, valid: np.ndarray | None = None) -> np.ndarray:
    finite = np.isfinite(values) if valid is None else (valid & np.isfinite(values))
    data = values[finite]
    if data.size == 0:
        return np.zeros((*values.shape, 3), np.uint8)
    if positive:
        lo, hi = np.percentile(data, [1, 99])
        idx = np.clip((values - lo) / max(hi - lo, 1e-14) * 255, 0, 255).astype(np.uint8)
        rgb = INFERNO[idx]
    else:
        lim = max(float(np.percentile(np.abs(data), 98)), 1e-14)
        idx = np.clip((values / lim * .5 + .5) * 255, 0, 255).astype(np.uint8)
        rgb = DIVERGING[idx]
    rgb[~finite] = (10, 13, 18)
    return np.ascontiguousarray(rgb)

