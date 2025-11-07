#!/usr/bin/env python3
"""
Dataset Lister Script
====================

This script lists all available dataset files in the path defined in the .env file.
It scans for common dataset file types and outputs their full paths.

Usage:
    python list_datasets.py
    python list_datasets.py --path /custom/path/to/datasets
    python list_datasets.py --extensions csv,json
    python list_datasets.py --no-recursive
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def load_dataset_path() -> Optional[str]:
    """
    Load the dataset path from the .env file.

    Returns:
        The dataset path if found, None otherwise
    """
    # Try to load environment variables from .env file
    if HAS_DOTENV:
        load_dotenv()
        dataset_path = os.getenv("DATASET_PATH")
    else:
        # Fallback: read .env file manually
        dataset_path = None
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DATASET_PATH="):
                        # Remove the variable name
                        dataset_path = line[12:].strip()
                        # Handle case where there might be an extra = at the beginning
                        if dataset_path.startswith("="):
                            dataset_path = dataset_path[1:].strip()
                        # Remove quotes if present
                        if (
                            dataset_path.startswith("'") and dataset_path.endswith("'")
                        ) or (
                            dataset_path.startswith('"') and dataset_path.endswith('"')
                        ):
                            dataset_path = dataset_path[1:-1]
                        break
        except FileNotFoundError:
            print("Error: .env file not found")
            return None

    if not dataset_path:
        print("Error: DATASET_PATH not found in .env file")
        return None

    return dataset_path


def find_dataset_files(
    directory: str, extensions: List[str], recursive: bool = True
) -> List[str]:
    """
    Find all dataset files in the given directory.

    Args:
        directory: The directory to search in
        extensions: List of file extensions to look for (without the dot)
        recursive: Whether to search recursively

    Returns:
        List of full paths to dataset files
    """
    dataset_files = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Error: Directory does not exist: {directory}")
        return dataset_files

    if not dir_path.is_dir():
        print(f"Error: Path is not a directory: {directory}")
        return dataset_files

    # Convert extensions to lowercase and ensure they don't start with a dot
    extensions = [ext.lower().lstrip(".") for ext in extensions]

    # Search for files
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            # Check if the file extension matches any of the desired extensions
            file_ext = file_path.suffix.lower().lstrip(".")
            if file_ext in extensions:
                dataset_files.append(str(file_path.absolute()))

    return sorted(dataset_files)


def main():
    """Main function to list dataset files."""
    parser = argparse.ArgumentParser(
        description="List dataset files in the specified directory"
    )

    parser.add_argument(
        "--path", help="Custom path to search for datasets (overrides .env file)"
    )

    parser.add_argument(
        "--extensions",
        help="Comma-separated list of file extensions to look for (without dots)",
        default="csv,json,xlsx,xls,parquet,pkl,feather,h5",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search recursively in subdirectories",
    )

    args = parser.parse_args()

    # Determine the dataset path
    if args.path:
        dataset_path = args.path
    else:
        dataset_path = load_dataset_path()
        if not dataset_path:
            sys.exit(1)

    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(",")]

    # Find dataset files
    dataset_files = find_dataset_files(
        dataset_path, extensions, recursive=not args.no_recursive
    )

    # Output results
    if dataset_files:
        print(f"Found {len(dataset_files)} dataset files in {dataset_path}:")
        for file_path in dataset_files:
            print(file_path)
    else:
        print(f"No dataset files found in {dataset_path}")
        print(f"Looked for files with extensions: {', '.join(extensions)}")


if __name__ == "__main__":
    main()
