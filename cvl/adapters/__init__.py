"""
Adapters for integrating external tools with CVlization patterns.

This module provides adapters for containerization tools like Cog,
enabling them to use CVlization's centralized caching and other patterns.
"""

from .cog import CogCacheAdapter

__all__ = ["CogCacheAdapter"]
