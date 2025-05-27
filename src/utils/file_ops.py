"""File and directory operation utilities."""

import os
from typing import Union
from pathlib import Path


def ensure_dir_exists(dir_path: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    This function creates the directory and any necessary parent directories
    if they don't already exist. If the directory already exists, no error
    is raised.

    Args:
        dir_path (Union[str, Path]): Path to the directory to create

    Raises:
        OSError: If the directory cannot be created due to permissions or other issues
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory '{dir_path}': {e}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path (Union[str, Path]): Path to the file

    Returns:
        int: File size in bytes

    Raises:
        FileNotFoundError: If the file doesn't exist
        OSError: If there's an error accessing the file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        return os.path.getsize(file_path)
    except OSError as e:
        raise OSError(f"Error getting size of file '{file_path}': {e}")


def list_files_with_extension(directory: Union[str, Path], extension: str) -> list[str]:
    """
    List all files in a directory with a specific extension.

    Args:
        directory (Union[str, Path]): Directory to search in
        extension (str): File extension to filter by (e.g., '.yaml', '.py')

    Returns:
        list[str]: List of file paths with the specified extension

    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Ensure extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension

    files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(extension.lower()):
            files.append(file_path)

    return sorted(files)
