"""
Shared pytest fixtures for Armature test suite.

This module provides common test fixtures and utilities used across all test modules.
Fixtures will be expanded in later phases as more functionality is added.
"""

from pathlib import Path

import pytest

from armature.classification import CategoryEnum
from armature.discovery import discover


@pytest.fixture
def temp_dir(tmp_path):
    """
    Provides a temporary directory for test file operations.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path object pointing to a temporary directory unique to this test
    """
    return tmp_path


def load_all_matrices():
    """Load all XES files with ground truth labels.

    Returns:
        Tuple of (matrices, labels) for all test data logs
    """
    TEST_DATA_ROOT = Path(__file__).parent.parent / "Test Data" / "Classification"

    matrices = []
    labels = []

    category_folders = {
        "structured": CategoryEnum.STRUCTURED,
        "semi-structured": CategoryEnum.SEMI_STRUCTURED,
        "loosely-structured": CategoryEnum.LOOSELY_STRUCTURED,
        "unstructured": CategoryEnum.UNSTRUCTURED,
    }

    for folder_name, category in category_folders.items():
        folder = TEST_DATA_ROOT / folder_name
        if not folder.exists():
            continue

        for xes_path in folder.glob("**/*.xes"):
            # Skip boundary cases (multi-label)
            if "looselyStructured_semiStructured" in xes_path.name:
                continue

            matrix = discover(xes_path)
            matrices.append(matrix)
            labels.append(category)

    return matrices, labels


# Additional fixtures will be added in later phases
