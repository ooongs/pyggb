"""
External interfaces for LLM and multimodal communication.

Provides interfaces for GPT-4V, Claude 3.5 Sonnet, and vLLM models with vision.
"""

from src.interfaces.multimodal_interface import MultimodalInterface, MultimodalMessage

__all__ = [
    "MultimodalInterface", "MultimodalMessage",
]

