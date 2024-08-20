def is_close_assert(actual, expected, abs_tol=0.01):
    assert abs(actual - expected) <= abs_tol, f"Assertion failed: Absolute difference between {actual} and {expected} exceeds {abs_tol}"
