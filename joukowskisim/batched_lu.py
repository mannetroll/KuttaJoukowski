"""Low-overhead parallel application of independent cached LU factors."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
from scipy import linalg


_CPU_COUNT = os.cpu_count() or 1
_WORKERS = max(
    1,
    min(4, int(os.environ.get("JOUKOWSKISIM_LU_WORKERS", _CPU_COUNT))),
)
_PARALLEL_COLUMNS = 192
_EXECUTOR = (
    ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="joukowski-lu")
    if _WORKERS > 1
    else None
)


def _column_chunks(count: int, workers: int) -> tuple[tuple[int, int], ...]:
    width = (count + workers - 1) // workers
    return tuple(
        (start, min(start + width, count))
        for start in range(0, count, width)
    )


def solve_cached_columns(
    factors: list[tuple[np.ndarray, np.ndarray]],
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve columns with distinct cached LU factors.

    SciPy's public ``lu_solve`` wrapper has appreciable Python overhead when
    called thousands of times per timestep.  Calling the matching LAPACK
    ``getrs`` routine directly and distributing coarse column ranges over a
    small persistent worker pool preserves the same factors and arithmetic
    while allowing independent solves to overlap.
    """
    values = np.asarray(rhs)
    if values.ndim != 2 or values.shape[1] != len(factors):
        raise ValueError("cached LU right-hand side has the wrong shape")
    if not factors:
        return np.empty_like(values)

    split_complex = (
        np.iscomplexobj(values)
        and not np.iscomplexobj(factors[0][0])
    )
    # Poisson matrices are real while their Fourier right-hand sides are
    # complex.  Passing both to get_lapack_funcs selects zgetrs and converts
    # every real LU matrix on every call.  Solve the real and imaginary parts
    # together as two dgetrs right-hand sides instead.
    getrs = linalg.lapack.get_lapack_funcs(
        "getrs",
        (factors[0][0],) if split_complex else (factors[0][0], values),
    )
    if split_complex:
        # Layout each modal pair as an F-contiguous (rows, 2) view without a
        # per-column allocation.  LAPACK then solves both Fourier components
        # in one dgetrs call and writes back into this shared workspace.
        work = np.empty(
            (values.shape[1], 2, values.shape[0]),
            dtype=factors[0][0].dtype,
        )
        work[:, 0, :] = values.real.T
        work[:, 1, :] = values.imag.T
        out = None
    else:
        work = None
        out = np.empty(
            values.shape,
            dtype=np.result_type(values.dtype, factors[0][0].dtype),
        )

    def solve_range(bounds: tuple[int, int]) -> None:
        start, stop = bounds
        for column in range(start, stop):
            lu, pivots = factors[column]
            if split_complex:
                _, info = getrs(
                    lu,
                    pivots,
                    work[column].T,
                    trans=0,
                    overwrite_b=True,
                )
            else:
                solution, info = getrs(
                    lu,
                    pivots,
                    # getrs overwrites its RHS.  Always own this column so a
                    # contiguous/Fortran caller cannot be mutated through a
                    # view, and promote real RHS for complex LU factors.
                    np.array(
                        values[:, column],
                        dtype=out.dtype,
                        order="C",
                        copy=True,
                    ),
                    trans=0,
                    overwrite_b=True,
                )
            if info != 0:
                raise RuntimeError(
                    f"cached LAPACK solve failed in column {column} "
                    f"with info={info}"
                )
            if not split_complex:
                out[:, column] = solution

    if _EXECUTOR is None or values.shape[1] < _PARALLEL_COLUMNS:
        solve_range((0, values.shape[1]))
    else:
        # Materialize the iterator so worker exceptions are raised here.
        tuple(
            _EXECUTOR.map(
                solve_range,
                _column_chunks(values.shape[1], _WORKERS),
            )
        )
    if split_complex:
        return np.ascontiguousarray(
            work[:, 0, :].T + 1j * work[:, 1, :].T
        )
    return np.ascontiguousarray(out)
