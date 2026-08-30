import numpy as np
from scipy import linalg

from joukowskisim.batched_lu import (
    _GROUP_CALL_WORK,
    _balanced_group_tasks,
    solve_cached_columns,
    solve_cached_groups,
)


def _factors(count: int, size: int):
    rng = np.random.default_rng(4895)
    matrices = rng.standard_normal((count, size, size))
    matrices += (size + 1.0) * np.eye(size)[None, :, :]
    return [
        linalg.lu_factor(matrix, check_finite=False)
        for matrix in matrices
    ]


def _sequential_groups(sizes):
    start = 0
    groups = []
    for size in sizes:
        groups.append(tuple(range(start, start + size)))
        start += size
    return tuple(groups)


def test_group_tasks_deterministically_balance_skewed_rhs_counts():
    sizes = (100, 56, 34, 30, 26, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2)
    groups = _sequential_groups(sizes)

    first = _balanced_group_tasks(groups, 4)
    second = _balanced_group_tasks(groups, 4)

    assert first == second
    assert sorted(group_index for task in first for group_index in task) == list(
        range(len(groups))
    )
    loads = tuple(
        sum(
            _GROUP_CALL_WORK + sizes[group_index]
            for group_index in task
        )
        for task in first
    )
    assert loads == (216, 220, 244, 210)


def test_balanced_group_tasks_preserve_real_and_split_complex_results():
    sizes = (100, 56, 34, 30, 26, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2)
    groups = _sequential_groups(sizes)
    factors = _factors(len(groups), 5)
    rng = np.random.default_rng(4815)
    real_rhs = rng.standard_normal((5, sum(sizes)))

    for rhs in (real_rhs, real_rhs + 1j * rng.standard_normal(real_rhs.shape)):
        expected = np.empty_like(rhs)
        for factor, group in zip(factors, groups):
            expected[:, group] = linalg.lu_solve(
                factor,
                rhs[:, group],
                check_finite=False,
            )

        original = rhs.copy()
        got = solve_cached_groups(factors, groups, rhs)

        assert np.allclose(got, expected, rtol=2e-13, atol=2e-13)
        assert np.array_equal(rhs, original)


def test_parallel_cached_lu_matches_scipy_for_real_columns():
    count, size = 200, 6  # Exercises the parallel-column path.
    factors = _factors(count, size)
    rhs = np.random.default_rng(17).standard_normal((size, count))
    expected = np.column_stack([
        linalg.lu_solve(factor, rhs[:, column], check_finite=False)
        for column, factor in enumerate(factors)
    ])

    original = rhs.copy()
    got = solve_cached_columns(factors, rhs)

    assert np.array_equal(got, expected)
    assert np.array_equal(rhs, original)


def test_parallel_real_lu_handles_complex_fourier_columns():
    count, size = 200, 6
    factors = _factors(count, size)
    rng = np.random.default_rng(23)
    rhs = rng.standard_normal((size, count)) + 1j * rng.standard_normal(
        (size, count)
    )
    expected = np.column_stack([
        linalg.lu_solve(factor, rhs[:, column], check_finite=False)
        for column, factor in enumerate(factors)
    ])

    got = solve_cached_columns(factors, rhs)

    assert np.allclose(got, expected, rtol=2e-13, atol=2e-13)


def test_complex_lu_promotes_real_rhs_without_mutating_it():
    count, size = 5, 4
    rng = np.random.default_rng(91)
    matrices = rng.standard_normal((count, size, size)).astype(complex)
    matrices += 1j * rng.standard_normal((count, size, size))
    matrices += (size + 1.0) * np.eye(size)[None, :, :]
    factors = [
        linalg.lu_factor(matrix, check_finite=False)
        for matrix in matrices
    ]
    rhs = np.asfortranarray(rng.standard_normal((size, count)))
    original = rhs.copy()
    expected = np.column_stack([
        linalg.lu_solve(factor, rhs[:, column], check_finite=False)
        for column, factor in enumerate(factors)
    ])

    got = solve_cached_columns(factors, rhs)

    assert np.iscomplexobj(got)
    assert np.allclose(got, expected, rtol=2e-13, atol=2e-13)
    assert np.array_equal(rhs, original)


def test_grouped_cached_lu_matches_individual_solves_without_mutating_rhs():
    count, size = 200, 6  # Exercises grouped multi-RHS and parallel paths.
    groups = tuple((column, count - 1 - column) for column in range(count // 2))
    factors = _factors(len(groups), size)
    rhs = np.asfortranarray(
        np.random.default_rng(31415).standard_normal((size, count))
    )
    expected = np.empty_like(rhs)
    for factor, group in zip(factors, groups):
        expected[:, group] = linalg.lu_solve(
            factor,
            rhs[:, group],
            check_finite=False,
        )

    original = rhs.copy()
    got = solve_cached_groups(factors, groups, rhs)

    assert np.array_equal(got, expected)
    assert np.array_equal(rhs, original)


def test_grouped_cached_lu_supports_mixed_group_sizes_and_c_order_rhs():
    size = 7
    groups = ((0, 7, 3), (1,), (2, 6), (4, 5))
    factors = _factors(len(groups), size)
    rhs = np.random.default_rng(2718).standard_normal((size, 8))
    expected = np.empty_like(rhs)
    for factor, group in zip(factors, groups):
        expected[:, group] = linalg.lu_solve(
            factor,
            rhs[:, group],
            check_finite=False,
        )

    original = rhs.copy()
    got = solve_cached_groups(factors, groups, rhs)

    assert np.array_equal(got, expected)
    assert np.array_equal(rhs, original)


def test_grouped_real_lu_preserves_split_complex_path():
    size = 6
    groups = ((0, 5), (1,), (2, 3, 4))
    factors = _factors(len(groups), size)
    rng = np.random.default_rng(1618)
    rhs = rng.standard_normal((size, 6)) + 1j * rng.standard_normal((size, 6))
    expected = np.empty_like(rhs)
    for factor, group in zip(factors, groups):
        expected[:, group] = linalg.lu_solve(
            factor,
            rhs[:, group],
            check_finite=False,
        )

    original = rhs.copy()
    got = solve_cached_groups(factors, groups, rhs)

    assert np.allclose(got, expected, rtol=2e-13, atol=2e-13)
    assert np.array_equal(rhs, original)
