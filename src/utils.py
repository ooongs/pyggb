#!/usr/bin/env python3
"""
Utility functions for path handling and project configuration.
Provides portable path resolution relative to project root.
"""

from pathlib import Path
from typing import Optional
import os


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory.
    """
    return Path(__file__).parent.parent.resolve()


def get_src_dir() -> Path:
    """
    Get the src directory.
    
    Returns:
        Path to the src directory.
    """
    return Path(__file__).parent.resolve()


def get_prompts_dir() -> Path:
    """
    Get the prompts directory.
    
    Returns:
        Path to the prompts directory.
    """
    return get_project_root() / "prompts"


def get_hints_dir() -> Path:
    """
    Get the hints directory within prompts.
    
    Returns:
        Path to the prompts/hints directory.
    """
    return get_prompts_dir() / "hints"


def get_examples_dir() -> Path:
    """
    Get the examples directory.
    
    Returns:
        Path to the examples directory.
    """
    return get_project_root() / "examples"


def get_data_dir() -> Path:
    """
    Get the data directory.
    
    Returns:
        Path to the data directory.
    """
    return get_project_root() / "data"


def get_data_dir() -> Path:
    """
    Get the ground truth directory.
    
    Returns:
        Path to the data directory.
    """
    return get_project_root() / "data"


def get_default_dataset_path() -> Path:
    """
    Get the default dataset path.
    
    Returns:
        Path to the default dataset file.
    """
    return get_data_dir() / "geoqa3_dataset.json"


def get_output_dir(subdir: Optional[str] = None) -> Path:
    """
    Get an output directory, creating it if necessary.
    
    Args:
        subdir: Optional subdirectory name (e.g., 'agent_logs', 'agent_images')
        
    Returns:
        Path to the output directory.
    """
    base = get_project_root()
    if subdir:
        output_path = base / subdir
    else:
        output_path = base / "output"
    
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_path(path: str, relative_to: Optional[Path] = None) -> Path:
    """
    Resolve a path, handling both absolute and relative paths.
    
    Args:
        path: Path string to resolve.
        relative_to: Base path for relative paths. Defaults to project root.
        
    Returns:
        Resolved absolute Path.
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    
    base = relative_to if relative_to else get_project_root()
    return (base / p).resolve()


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists.
        
    Returns:
        The same path, guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# For backwards compatibility with os.path style code
def join_from_root(*parts: str) -> str:
    """
    Join path parts relative to project root.
    
    Args:
        *parts: Path components to join.
        
    Returns:
        Absolute path string.
    """
    return str(get_project_root().joinpath(*parts))

