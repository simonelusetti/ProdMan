"""Expose the product explorer entry point under its own module."""

from .explorer import explorer as product_explorer

__all__ = ["product_explorer"]
