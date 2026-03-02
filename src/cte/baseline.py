"""Compatibility surface for the parity baseline implementation.

This module exposes the v1 pipelines used in Gate 1/Gate 1A.
"""

from .pipelines.no_rag import v1 as no_rag_v1
from .pipelines.rag import v1 as rag_v1

__all__ = ["no_rag_v1", "rag_v1"]
