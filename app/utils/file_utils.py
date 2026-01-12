"""File I/O utilities"""

import os
from pathlib import Path
from typing import Optional


def read_input_file(file_path: str) -> Optional[str]:
    """
    Read requirements from input file
    
    Args:
        file_path: Path to input file
        
    Returns:
        File content as string, or None if file not found
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ File {file_path} not found")
        print(f"💡 Please create {file_path} and paste client requirements there")
        return None
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return None


def ensure_output_dir(directory: str) -> Path:
    """
    Ensure output directory exists, create if not
    
    Args:
        directory: Directory path
        
    Returns:
        Path object to the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

