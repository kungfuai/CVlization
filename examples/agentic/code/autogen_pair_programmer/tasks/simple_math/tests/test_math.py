import math

import pytest

from tasks.simple_math.task import sum_of_squares


def test_positive_values():
    assert sum_of_squares([1, 2, 3]) == 14


def test_negative_values():
    assert sum_of_squares([-2, -1, 3]) == 14


def test_empty_iterable():
    assert sum_of_squares([]) == 0


def test_floats():
    assert math.isclose(sum_of_squares([0.5, 1.5]), 2.5)


def test_invalid_element():
    with pytest.raises(TypeError):
        sum_of_squares([1, "oops", 2])
