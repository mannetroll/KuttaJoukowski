import numpy as np
from scipy import linalg

from joukowskisim.batched_lu import solve_cached_columns


def _factors(count: int, size: int):
    rng = np.random.default_rng(4895)
    matrices = rng.standard_normal((count, size, size))
    matrices += (size + 1.0) * np.eye(size)[None, :, :]
    return [
        linalg.lu_factor(matrix, check_finite=False)
        for matrix in matrices
    ]


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
