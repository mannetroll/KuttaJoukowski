"""Low-overhead parallel application of independent cached LU factors."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

import numpy as np
from scipy import linalg


_CPU_COUNT = os.cpu_count() or 1
_WORKERS = max(
    1,
    min(4, int(os.environ.get("JOUKOWSKISIM_LU_WORKERS", _CPU_COUNT))),
)
_PARALLEL_COLUMNS = 192
# A grouped solve has fixed Python/LAPACK dispatch and scatter costs in
# addition to work proportional to its RHS count.  Production-size (Nr=240)
# microbenchmarks put that fixed component near 32 column-equivalents.  This
# weight outperformed both equal-group chunks and RHS-count-only balancing.
_GROUP_CALL_WORK = 32
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


@lru_cache(maxsize=32)
def _balanced_group_tasks(
    column_groups: tuple[tuple[int, ...], ...],
    workers: int,
) -> tuple[tuple[int, ...], ...]:
    """Assign variable-size LU groups to workers with deterministic LPT.

    A grouped ``getrs`` solve has both a fixed call cost and work proportional
    to its number of right-hand-side columns.  Contiguous equal-count chunks
    can therefore be badly imbalanced when profile grouping produces a few
    large groups.  The longest-processing-time greedy schedule assigns the
    largest estimated jobs first to the currently lightest worker.  Stable
    index tie-breaks keep the schedule reproducible, and caching removes the
    scheduling overhead from repeated preconditioner applications.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if not column_groups:
        return ()

    task_count = min(workers, len(column_groups))
    assignments: list[list[int]] = [[] for _ in range(task_count)]
    loads = [0] * task_count
    group_indices = sorted(
        range(len(column_groups)),
        key=lambda group_index: (
            -(_GROUP_CALL_WORK + len(column_groups[group_index])),
            group_index,
        ),
    )
    for group_index in group_indices:
        worker = min(
            range(task_count),
            key=lambda worker_index: (loads[worker_index], worker_index),
        )
        assignments[worker].append(group_index)
        loads[worker] += _GROUP_CALL_WORK + len(column_groups[group_index])
    return tuple(tuple(assignment) for assignment in assignments)


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


def solve_cached_groups(
    factors: list[tuple[np.ndarray, np.ndarray]],
    column_groups: tuple[tuple[int, ...], ...],
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve groups of columns that share one cached LU factor.

    Each group is passed to LAPACK as a multi-right-hand-side solve.  This is
    useful when symmetry makes several independent column matrices identical:
    it reduces both factor storage and Python/LAPACK call overhead without
    changing any right-hand side.  Groups must form a partition of the RHS
    columns, and the caller's array is never overwritten.
    """
    values = np.asarray(rhs)
    if values.ndim != 2 or len(factors) != len(column_groups):
        raise ValueError("cached LU groups have the wrong shape")
    if not factors:
        if values.shape[1] != 0:
            raise ValueError("cached LU groups do not cover the RHS columns")
        return np.empty_like(values)

    columns = tuple(column for group in column_groups for column in group)
    if (
        any(not group for group in column_groups)
        or len(columns) != values.shape[1]
        or tuple(sorted(columns)) != tuple(range(values.shape[1]))
    ):
        raise ValueError("cached LU groups must partition the RHS columns")

    split_complex = (
        np.iscomplexobj(values)
        and not np.iscomplexobj(factors[0][0])
    )
    getrs = linalg.lapack.get_lapack_funcs(
        "getrs",
        (factors[0][0],) if split_complex else (factors[0][0], values),
    )
    out = np.empty(
        values.shape,
        dtype=np.result_type(values.dtype, factors[0][0].dtype),
    )
    if _EXECUTOR is None or values.shape[1] < _PARALLEL_COLUMNS:
        group_tasks = (tuple(range(len(column_groups))),)
    else:
        group_tasks = _balanced_group_tasks(column_groups, _WORKERS)
    # Real-valued stages dominate the mapped implicit preconditioner.  Give
    # each parallel task one reusable C-order scratch slab; transposing a
    # populated (group, rows) slice gives LAPACK the F-contiguous
    # (rows, group) multi-RHS matrix it expects.  This avoids allocating and
    # freeing one temporary array for every group while keeping worker writes
    # disjoint.  The split-complex path retains its paired real/imag layout.
    real_workspace = (
        None
        if split_complex
        else np.empty(
            (
                len(group_tasks),
                max(len(group) for group in column_groups),
                values.shape[0],
            ),
            dtype=out.dtype,
            order="C",
        )
    )

    def solve_task(task: tuple[int, tuple[int, ...]]) -> None:
        workspace_index, group_indices = task
        scratch = (
            None
            if real_workspace is None
            else real_workspace[workspace_index]
        )
        for group_index in group_indices:
            group = column_groups[group_index]
            lu, pivots = factors[group_index]
            if split_complex:
                group_size = len(group)
                work = np.empty(
                    (values.shape[0], 2 * group_size),
                    dtype=lu.dtype,
                    order="F",
                )
                work[:, :group_size] = values[:, group].real
                work[:, group_size:] = values[:, group].imag
                solution, info = getrs(
                    lu,
                    pivots,
                    work,
                    trans=0,
                    overwrite_b=True,
                )
                if info == 0:
                    out[:, group] = (
                        solution[:, :group_size]
                        + 1j * solution[:, group_size:]
                    )
            else:
                group_size = len(group)
                for local_column, source_column in enumerate(group):
                    scratch[local_column] = values[:, source_column]
                solution, info = getrs(
                    lu,
                    pivots,
                    scratch[:group_size].T,
                    trans=0,
                    overwrite_b=True,
                )
                if info == 0:
                    for local_column, destination_column in enumerate(group):
                        out[:, destination_column] = solution[:, local_column]
            if info != 0:
                raise RuntimeError(
                    f"cached LAPACK solve failed in group {group_index} "
                    f"with info={info}"
                )

    tasks = tuple(enumerate(group_tasks))
    if len(tasks) == 1:
        solve_task(tasks[0])
    else:
        # Materialize the iterator so worker exceptions are raised here.
        tuple(_EXECUTOR.map(solve_task, tasks))
    return np.ascontiguousarray(out)
