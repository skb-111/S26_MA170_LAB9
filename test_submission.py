"""
Test cases for Lab 09 Real - DO NOT MODIFY
"""

import torch
from submission import Vector2D, batch_right_multiply_einsum


def test_vector2d_stores_attributes():
    v = Vector2D(3, -1)
    assert v.x == 3
    assert v.y == -1


def test_vector2d_str():
    v = Vector2D(2, 5)
    assert str(v) == "Vector2D(2, 5)"


def test_vector2d_add_returns_new_vector():
    v1 = Vector2D(1, 2)
    v2 = Vector2D(-3, 4)
    result = v1.add(v2)
    assert isinstance(result, Vector2D)
    assert result.x == -2
    assert result.y == 6


def test_vector2d_dot():
    v1 = Vector2D(1, 2)
    v2 = Vector2D(3, 4)
    assert v1.dot(v2) == 11


def test_batch_right_multiply_einsum_shape():
    T = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    B = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    result = batch_right_multiply_einsum(T, B)
    assert result.shape == (2, 3, 5)


def test_batch_right_multiply_einsum_matches_slice_by_slice_matmul():
    T = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    B = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [1.0, 2.0, 3.0],
        ]
    )
    result = batch_right_multiply_einsum(T, B)
    expected = torch.stack([T[0] @ B, T[1] @ B], dim=0)
    assert torch.equal(result, expected)


def test_batch_right_multiply_einsum_batch_values():
    T = torch.tensor(
        [
            [[1.0, 0.0, 2.0]],
            [[0.0, 1.0, 1.0]],
        ]
    )
    B = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    result = batch_right_multiply_einsum(T, B)
    expected = torch.tensor(
        [
            [[11.0, 14.0]],
            [[8.0, 10.0]],
        ]
    )
    assert torch.equal(result, expected)
