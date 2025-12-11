"""
PyGGB - Python Geometry Problem Solving with GeoGebra DSL

This package provides tools for:
- Geometry construction via DSL
- Problem parsing and validation
- Agent-based problem solving
- Benchmark evaluation
"""

from src.utils import get_project_root, get_prompts_dir, get_data_dir

__version__ = "0.1.0"
__all__ = [
    "core",
    "dsl",
    "agent",
    "benchmark",
    "interfaces",
    "parser",
    "ggb",
]

