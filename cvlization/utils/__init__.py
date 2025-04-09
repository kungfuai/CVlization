"""General utility functions."""

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
from .object import getattr_recursively, setattr_recursively
