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
_COLUMN_WORKERS = max(
    1,
    min(
        8,
        int(
            os.environ.get(
                "JOUKOWSKISIM_POISSON_LU_WORKERS",
                os.environ.get("JOUKOWSKISIM_LU_WORKERS", _CPU_COUNT),
            )
        ),
    ),
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
_COLUMN_EXECUTOR = (
    ThreadPoolExecutor(
        max_workers=_COLUMN_WORKERS,
        thread_name_prefix="joukowski-poisson-lu",
    )
    if _COLUMN_WORKERS > 1
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


@lru_cache(maxsize=32)
def _group_layout(
    column_groups: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache grouped column order, its inverse, and group boundaries."""
    columns = tuple(column for group in column_groups for column in group)
    if (
        any(not group for group in column_groups)
        or tuple(sorted(columns)) != tuple(range(len(columns)))
    ):
        raise ValueError("cached LU groups must partition the RHS columns")
    permutation = np.fromiter(columns, dtype=np.intp, count=len(columns))
    inverse = np.argsort(permutation)
    offsets = np.empty(len(column_groups) + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(
        np.fromiter(
            (len(group) for group in column_groups),
            dtype=np.intp,
            count=len(column_groups),
        ),
        out=offsets[1:],
    )
    for layout in (permutation, inverse, offsets):
        layout.flags.writeable = False
    return permutation, inverse, offsets


class GroupedRealLUPlan:
    """Compiled application plan for one set of grouped real LU factors.

    The radial IMEX preconditioner applies the same stage factors hundreds of
    times between timestep-cache rebuilds.  Resolve its permutation, LAPACK
    function, factor lookups, and balanced task partition once.  At production
    widths the persistent worker pool handles one partition per worker while
    the calling thread handles one additional partition instead of waiting
    idle for all grouped solves to finish.

    This specialized plan accepts real right-hand sides and real LU factors.
    :func:`solve_cached_groups` remains the generic entry point for serial,
    complex-factor, and split-complex uses.
    """

    def __init__(
        self,
        factors: list[tuple[np.ndarray, np.ndarray]],
        column_groups: tuple[tuple[int, ...], ...],
    ):
        if not factors or len(factors) != len(column_groups):
            raise ValueError("cached LU groups have the wrong shape")
        if np.iscomplexobj(factors[0][0]):
            raise ValueError("compiled grouped LU factors must be real")

        permutation, inverse, offsets = _group_layout(column_groups)
        if _EXECUTOR is None or permutation.size < _PARALLEL_COLUMNS:
            group_tasks = (tuple(range(len(column_groups))),)
        else:
            # Keep one task for every persistent worker and one for the
            # calling thread.  The caller formerly waited idle in map().
            group_tasks = _balanced_group_tasks(
                column_groups,
                _WORKERS + 1,
            )

        self._permutation = permutation
        self._inverse = inverse
        self._factor_dtype = factors[0][0].dtype
        self._getrs = linalg.lapack.get_lapack_funcs(
            "getrs",
            (factors[0][0],),
        )
        self._job_tasks = tuple(
            tuple(
                (
                    factors[group_index][0],
                    factors[group_index][1],
                    int(offsets[group_index]),
                    int(offsets[group_index + 1]),
                )
                for group_index in group_indices
            )
            for group_indices in group_tasks
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Apply the compiled factors without mutating ``rhs``."""
        values = np.asarray(rhs)
        if (
            values.ndim != 2
            or values.shape[1] != self._permutation.size
            or np.iscomplexobj(values)
        ):
            raise ValueError(
                "compiled grouped LU right-hand side must be a real "
                "matrix with the planned column count"
            )

        ordered = np.asfortranarray(
            values[:, self._permutation],
            dtype=np.result_type(values.dtype, self._factor_dtype),
        )

        def solve_task(jobs) -> None:
            for lu, pivots, start, stop in jobs:
                _, info = self._getrs(
                    lu,
                    pivots,
                    ordered[:, start:stop],
                    trans=0,
                    overwrite_b=True,
                )
                if info != 0:
                    raise RuntimeError(
                        "cached LAPACK solve failed in compiled group "
                        f"[{start}:{stop}] with info={info}"
                    )

        if len(self._job_tasks) == 1:
            solve_task(self._job_tasks[0])
        else:
            # There are at most _WORKERS submitted tasks; the final balanced
            # partition runs concurrently on the calling thread.  Await every
            # future so worker failures are propagated before returning.
            futures = tuple(
                _EXECUTOR.submit(solve_task, jobs)
                for jobs in self._job_tasks[:-1]
            )
            caller_error = None
            try:
                solve_task(self._job_tasks[-1])
            except Exception as error:  # Preserve it after joining workers.
                caller_error = error
            for future in futures:
                future.result()
            if caller_error is not None:
                raise caller_error

        return np.ascontiguousarray(ordered[:, self._inverse])


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

    if _COLUMN_EXECUTOR is None or values.shape[1] < _PARALLEL_COLUMNS:
        solve_range((0, values.shape[1]))
    else:
        # Materialize the iterator so worker exceptions are raised here.
        tuple(
            _COLUMN_EXECUTOR.map(
                solve_range,
                _column_chunks(values.shape[1], _COLUMN_WORKERS),
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

    permutation, inverse, offsets = _group_layout(column_groups)
    if permutation.size != values.shape[1]:
        raise ValueError("cached LU groups must partition the RHS columns")

    split_complex = (
        np.iscomplexobj(values)
        and not np.iscomplexobj(factors[0][0])
    )
    getrs = linalg.lapack.get_lapack_funcs(
        "getrs",
        (factors[0][0],) if split_complex else (factors[0][0], values),
    )
    out_dtype = np.result_type(values.dtype, factors[0][0].dtype)
    if _EXECUTOR is None or values.shape[1] < _PARALLEL_COLUMNS:
        group_tasks = (tuple(range(len(column_groups))),)
    else:
        group_tasks = _balanced_group_tasks(column_groups, _WORKERS)
    # Real-valued stages dominate the mapped implicit preconditioner.  Gather
    # all groups once into adjacent columns of one Fortran-order array.  Each
    # LAPACK call can then overwrite a direct disjoint slice, avoiding the
    # per-column Python gather/scatter loops formerly paid for every group.
    ordered = (
        None
        if split_complex
        else np.asfortranarray(values[:, permutation], dtype=out_dtype)
    )
    out = None if not split_complex else np.empty(values.shape, dtype=out_dtype)

    def solve_task(group_indices: tuple[int, ...]) -> None:
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
                start = int(offsets[group_index])
                stop = int(offsets[group_index + 1])
                _, info = getrs(
                    lu,
                    pivots,
                    ordered[:, start:stop],
                    trans=0,
                    overwrite_b=True,
                )
            if info != 0:
                raise RuntimeError(
                    f"cached LAPACK solve failed in group {group_index} "
                    f"with info={info}"
                )

    if len(group_tasks) == 1:
        solve_task(group_tasks[0])
    else:
        # Materialize the iterator so worker exceptions are raised here.
        tuple(_EXECUTOR.map(solve_task, group_tasks))
    if split_complex:
        return np.ascontiguousarray(out)
    return np.ascontiguousarray(ordered[:, inverse])
