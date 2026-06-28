"""
Smoke tests for Armature package.

These tests verify that the package is properly installed and importable,
without testing specific functionality (which will be added in later phases).
"""

import pytest


def test_armature_imports():
    """Test that the armature package can be imported without errors."""
    try:
        import armature
    except ImportError as e:
        pytest.fail(f"Failed to import armature package: {e}")


def test_armature_version():
    """Test that armature package has the expected version."""
    import armature
    
    assert hasattr(armature, "__version__"), "armature package missing __version__ attribute"
    assert armature.__version__ == "0.1.0", (
        f"Expected version 0.1.0, got {armature.__version__}"
    )


def test_submodules_importable():
    """Test that all 5 submodules can be imported without errors."""
    submodules = ["core", "discovery", "classification", "cli", "web"]
    
    for submodule in submodules:
        try:
            module = __import__(f"armature.{submodule}", fromlist=[submodule])
        except ImportError as e:
            pytest.fail(f"Failed to import armature.{submodule}: {e}")


def test_submodules_have_docstrings():
    """Test that each submodule has a docstring describing its purpose."""
    submodules = ["core", "discovery", "classification", "cli", "web"]
    
    for submodule in submodules:
        module = __import__(f"armature.{submodule}", fromlist=[submodule])
        assert hasattr(module, "__doc__"), (
            f"armature.{submodule} missing __doc__ attribute"
        )
        assert module.__doc__ is not None and len(module.__doc__.strip()) > 0, (
            f"armature.{submodule} has empty docstring"
        )
