"""cdrift-evaluation benchmark dataset adapter.

Provides CdriftDataset class for enumerating benchmark logs and extracting
ground truth changepoints from XES metadata or filename patterns.

Usage:
    dataset = CdriftDataset(Path("cdrift-evaluation"))
    for log_info in dataset.list_logs():
        print(log_info["path"], log_info["ground_truth"])
"""
from pathlib import Path
from typing import List, Optional
import re
import logging
import ast
import pandas as pd

logger = logging.getLogger(__name__)


class CdriftDataset:
    """Adapter for cdrift-evaluation benchmark logs.

    Enumerates XES logs in EvaluationLogs/ and extracts ground truth changepoints.
    """

    def __init__(self, cdrift_path: Path):
        """Initialize dataset adapter.

        Args:
            cdrift_path: Path to cloned cdrift-evaluation repository
        """
        self.cdrift_path = Path(cdrift_path)
        self.eval_logs_path = self.cdrift_path / "EvaluationLogs"

        if not self.eval_logs_path.exists():
            logger.warning(f"EvaluationLogs directory not found: {self.eval_logs_path}")

        # Load ground truth CSV
        csv_path = self.cdrift_path / "algorithm_results.csv"
        if csv_path.exists():
            try:
                self.ground_truth_df = pd.read_csv(csv_path)
                logger.debug(f"Loaded ground truth CSV with {len(self.ground_truth_df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load ground truth CSV from {csv_path}: {e}")
                self.ground_truth_df = None
        else:
            logger.warning(f"Ground truth CSV not found: {csv_path}")
            self.ground_truth_df = None

    def list_logs(self) -> List[Path]:
        """Enumerate all XES logs in benchmark.

        Returns:
            List of paths to .xes and .xes.gz files
        """
        if not self.eval_logs_path.exists():
            return []

        # Find all .xes and .xes.gz files
        xes_files = list(self.eval_logs_path.glob("**/*.xes*"))

        return sorted(xes_files)

    def list_logs_by_source(self, source: str) -> List[Path]:
        """Filter logs by source (Bose/Ceravolo/Ostovar).

        Args:
            source: Source name (Bose, Ceravolo, or Ostovar)

        Returns:
            List of paths matching source
        """
        all_logs = self.list_logs()
        source_dir = source.lower()

        return [log_path for log_path in all_logs if source_dir in str(log_path.parent).lower()]

    def get_log_info(self, log_path: Path) -> dict:
        """Get log metadata including ground truth.

        Args:
            log_path: Path to XES log file

        Returns:
            Dict with log_source, log_name, ground_truth
        """
        # Extract source from parent directory
        source = log_path.parent.name

        # Extract log name (filename without extension)
        log_name = log_path.stem

        # Extract ground truth
        ground_truth = extract_ground_truth(log_path, dataset=self)

        return {
            "path": log_path,
            "log_source": source,
            "log_name": log_name,
            "ground_truth": ground_truth,
        }


def extract_ground_truth(log_path: Path, dataset: Optional["CdriftDataset"] = None) -> List[int]:
    """Extract ground truth changepoints from log.

    Tries multiple methods:
    1. CSV lookup (algorithm_results.csv)
    2. XES attributes (concept:drift, drift:changepoints)
    3. Filename pattern (log_cp_100_300.xes)

    Args:
        log_path: Path to XES log file
        dataset: Optional CdriftDataset instance with loaded CSV

    Returns:
        List of changepoint indices (trace numbers), empty if none found
    """
    # Try CSV lookup first
    if dataset and dataset.ground_truth_df is not None:
        # Strip .xes or .xes.gz extension to get log name
        log_name = log_path.name
        if log_name.endswith(".xes.gz"):
            log_name = log_name[:-7]  # Remove .xes.gz
        elif log_name.endswith(".xes"):
            log_name = log_name[:-4]  # Remove .xes

        try:
            # Look up log in CSV
            df = dataset.ground_truth_df
            log_row = df[df["Log"] == log_name]

            if not log_row.empty:
                # Parse changepoints from CSV column
                cp_str = log_row.iloc[0]["Actual Changepoints for Log"]
                if pd.notna(cp_str):
                    # Use ast.literal_eval to safely parse "[1199, 2399, ...]"
                    changepoints = ast.literal_eval(cp_str)
                    logger.debug(f"Extracted ground truth from CSV for {log_name}: {changepoints}")
                    return changepoints
        except Exception as e:
            logger.debug(f"CSV lookup failed for {log_name}: {e}")

    # Try XES attributes
    try:
        import pm4py

        log = pm4py.read_xes(str(log_path), return_legacy_log_object=True)

        # Check for drift-related attributes
        if hasattr(log, "attributes"):
            attrs = log.attributes

            # Look for common drift attribute names
            for attr_name in ["concept:drift", "drift:changepoints", "changepoints"]:
                if attr_name in attrs:
                    drift_value = attrs[attr_name]
                    # Parse changepoint list from attribute
                    changepoints = _parse_changepoint_attr(drift_value)
                    if changepoints:
                        logger.debug(
                            f"Extracted ground truth from XES attribute {attr_name}: {changepoints}"
                        )
                        return changepoints
    except Exception as e:
        logger.debug(f"Could not read XES attributes from {log_path}: {e}")

    # Fallback: parse filename pattern
    # Example: "log_cp_100_300.xes" → [100, 300]
    # Example: "log_cp_500.xes" → [500]
    filename = log_path.stem
    match = re.search(r"_cp_(\d+(?:_\d+)*)", filename)
    if match:
        cp_str = match.group(1)
        changepoints = [int(x) for x in cp_str.split("_")]
        logger.debug(f"Extracted ground truth from filename: {changepoints}")
        return changepoints

    # No ground truth found
    logger.debug(f"No ground truth found for {log_path}")
    return []


def _parse_changepoint_attr(attr_value) -> List[int]:
    """Parse changepoint attribute value to list of integers.

    Handles various formats:
    - "100,300"
    - "[100, 300]"
    - "100 300"

    Args:
        attr_value: Attribute value from XES

    Returns:
        List of changepoint indices
    """
    if not attr_value:
        return []

    # Convert to string
    value_str = str(attr_value).strip()

    # Remove brackets if present
    value_str = value_str.strip("[]")

    # Try comma-separated
    if "," in value_str:
        try:
            return [int(x.strip()) for x in value_str.split(",")]
        except ValueError:
            pass

    # Try space-separated
    try:
        return [int(x) for x in value_str.split() if x.strip()]
    except ValueError:
        pass

    # Try single value
    try:
        return [int(value_str)]
    except ValueError:
        pass

    return []


def list_benchmark_logs(cdrift_path: Path) -> List[dict]:
    """Convenience function to list all benchmark logs with metadata.

    Args:
        cdrift_path: Path to cdrift-evaluation repository

    Returns:
        List of dicts with path, source, name, ground_truth
    """
    dataset = CdriftDataset(cdrift_path)
    return [
        {
            "path": log_info["path"],
            "source": log_info["log_source"],
            "name": log_info["log_name"],
            "ground_truth": log_info["ground_truth"],
        }
        for log_info in [dataset.get_log_info(log_path) for log_path in dataset.list_logs()]
    ]
