"""
MATH 170 - Lab 09 Real: OOP and Einsum
Spring 2026

Instructions:
1. Implement a small Vector2D class using basic OOP.
2. Use torch.einsum for a batched tensor-matrix multiplication.
"""

import torch


class Vector2D:
    """
    Simple 2D vector class.

    Attributes
    ----------
    x : float
        First coordinate.
    y : float
        Second coordinate.
    """

    def __init__(self, x, y):
        """Store the coordinates on the instance."""
        # TODO: store x and y as instance attributes
        self.x = x
        self.y = y

    def __str__(self):
        """Return a readable string like Vector2D(3, 4)."""
        # TODO: return the string representation
        return f"Vector2D({self.x}, {self.y})"

    def add(self, other):
        """
        Return a new Vector2D equal to self + other.

        Parameters
        ----------
        other : Vector2D
            Another vector.
        """
        # TODO: return a new Vector2D with added coordinates
        return Vector2D(self.x + other.x, self.y + other.y)

    def dot(self, other):
        """
        Return the dot product self.x * other.x + self.y * other.y.

        Parameters
        ----------
        other : Vector2D
            Another vector.
        """
        # TODO: return the dot product
        return self.x * other.x + self.y * other.y


def batch_right_multiply_einsum(T: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Return the batched matrix product C using torch.einsum.

    Think of T as a batch of matrices:

        [A_0, A_1, ..., A_{b-1}]

    If T has shape (b, m, n), then each A_i = T[i, :, :] has shape (m, n).
    Let B have shape (n, k).

    Return a tensor C with shape (b, m, k) such that

        C[i, :, :] = A_i @ B

    for every batch index i.

    Use torch.einsum.
    """
    # TODO: use torch.einsum for batched right multiplication
    return torch.einsum('bmn,nk->bmk', T, B)
