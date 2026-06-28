"""
Root-level pytest configuration.

Ensures the armature package is importable during test runs without requiring
formal installation. This allows tests to run in development environments.
"""

import sys
from pathlib import Path

# Add the package root to Python path so 'import armature' works
package_root = Path(__file__).parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))
