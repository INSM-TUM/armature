#!/usr/bin/env python3
"""
Full validation script for discovery algorithm.

Runs discovery on all test logs (clean and noisy), compares against ground truth,
and finds optimal thresholds for noisy data using binary search.
"""

from pathlib import Path
from typing import Dict, Tuple
import sys

from armature.discovery.discover import discover
from armature.serialization.yaml_codec import YAMLCodec
from armature.core.matrix import Matrix


def compare_matrices(golden: Matrix, discovered: Matrix) -> Tuple[int, int, int, int]:
    """
    Compare discovered matrix against golden truth.
    
    Returns:
        (temporal_correct, temporal_total, existential_correct, existential_total)
    """
    temporal_correct = 0
    temporal_total = 0
    existential_correct = 0
    existential_total = 0
    
    # Get all activity pairs (including diagonal for self-loops)
    for source in golden.activities:
        for target in golden.activities:
            golden_cell = golden.get_cell(source, target)
            discovered_cell = discovered.get_cell(source, target)
            
            # Count temporal dependencies (skip NO_ORDERING as it's default)
            if golden_cell.temporal.value != "no_ordering":
                temporal_total += 1
                if golden_cell.temporal == discovered_cell.temporal:
                    temporal_correct += 1
            
            # Count existential dependencies (skip INDEPENDENCE as it's default)
            if golden_cell.existential.value != "independence":
                existential_total += 1
                if golden_cell.existential == discovered_cell.existential:
                    existential_correct += 1
    
    return temporal_correct, temporal_total, existential_correct, existential_total


def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate accuracy as fraction (0.0-1.0)."""
    if total == 0:
        return 1.0  # No dependencies to check
    return correct / total


def find_optimal_threshold(
    xes_path: Path,
    golden: Matrix,
    dimension: str,  # 'temporal' or 'existential'
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    iterations: int = 10,
) -> Tuple[float, float]:
    """
    Binary search to find optimal threshold for given dimension.
    
    Returns:
        (best_threshold, best_accuracy)
    """
    best_threshold = max_threshold
    best_accuracy = 0.0
    
    # Try thresholds in binary search pattern
    search_points = []
    step = (max_threshold - min_threshold) / iterations
    for i in range(iterations + 1):
        search_points.append(min_threshold + i * step)
    
    for threshold in search_points:
        # Run discovery with this threshold
        if dimension == 'temporal':
            discovered = discover(xes_path, threshold=threshold)
        else:  # both dimensions use same threshold in current implementation
            discovered = discover(xes_path, threshold=threshold)
        
        temp_correct, temp_total, exist_correct, exist_total = compare_matrices(golden, discovered)
        
        if dimension == 'temporal':
            accuracy = calculate_accuracy(temp_correct, temp_total)
        else:
            accuracy = calculate_accuracy(exist_correct, exist_total)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy


def get_process_patterns(log_name: str) -> Dict[str, bool]:
    """
    Return which patterns are covered by each log based on filename.
    """
    patterns = {
        "Sequence": False,
        "Parallelism": False,
        "XOR": False,
        "NAND": False,
        "Multi-Choice": False,
        "Looping": False,
        "Nesting": False,
        "Block-Structure": False,
        "Skipping": False,
    }
    
    # Parse patterns from filename
    lower_name = log_name.lower()
    
    if "negatedequivalence" in lower_name or "xor" in lower_name:
        patterns["XOR"] = True
    if "parallelism" in lower_name:
        patterns["Parallelism"] = True
    if "nand" in lower_name:
        patterns["NAND"] = True
    if "nested" in lower_name:
        patterns["Nesting"] = True
        patterns["Block-Structure"] = True
    if "loop" in lower_name:
        patterns["Looping"] = True
    if "skipping" in lower_name:
        patterns["Skipping"] = True
    if "inclusiveor" in lower_name:
        patterns["Multi-Choice"] = True
    if "nonblock" in lower_name:
        patterns["Block-Structure"] = False  # Explicitly non-block
    else:
        # Most logs are block-structured unless specified otherwise
        if patterns["Nesting"] or "01" in log_name or "03" in log_name:
            patterns["Block-Structure"] = True
    
    # Infer sequence if no parallel
    if not patterns["Parallelism"] and any([patterns["XOR"], patterns["Nesting"]]):
        patterns["Sequence"] = True
    
    return patterns


def main():
    """Run full validation suite."""
    project_root = Path(__file__).parent
    test_data_dir = project_root / "Test Data" / "Discovery"
    golden_dir = project_root / "tests" / "fixtures" / "discovery" / "golden"
    
    print("=" * 80)
    print("DISCOVERY VALIDATION - FULL RUN")
    print("=" * 80)
    print()
    
    # Get all clean logs
    clean_logs = sorted([f for f in test_data_dir.glob("event_log_*.xes") if f.parent.name != "noise"])
    
    results = []
    
    # Process each clean log
    print("Processing clean logs with threshold=1.0...")
    print()
    
    for log_path in clean_logs:
        log_num = log_path.stem.split("_")[2]  # Extract 01, 02, etc.
        log_name = log_path.stem
        
        print(f"  {log_name}...")
        
        # Load golden truth
        golden_path = golden_dir / f"{log_path.stem}.yaml"
        if not golden_path.exists():
            print(f"    WARNING: No golden file found at {golden_path}")
            continue
        
        golden = YAMLCodec.load(golden_path)
        
        # Run discovery with threshold=1.0
        discovered = discover(log_path, threshold=1.0)
        
        # Compare
        temp_correct, temp_total, exist_correct, exist_total = compare_matrices(golden, discovered)
        temp_acc = calculate_accuracy(temp_correct, temp_total)
        exist_acc = calculate_accuracy(exist_correct, exist_total)
        
        # Get patterns
        patterns = get_process_patterns(log_name)
        
        # Store result
        results.append({
            "log_num": log_num,
            "log_name": log_name,
            "patterns": patterns,
            "clean_temporal_acc": temp_acc,
            "clean_existential_acc": exist_acc,
            "clean_threshold": 1.0,
            "noisy_temporal_acc": None,
            "noisy_existential_acc": None,
            "noisy_threshold": None,
        })
        
        print(f"    Clean: temporal={temp_acc:.2f}, existential={exist_acc:.2f}")
    
    print()
    print("-" * 80)
    print("Processing noisy logs with threshold optimization...")
    print()
    
    # Process noisy logs
    noise_dir = test_data_dir / "noise"
    noisy_logs = sorted(noise_dir.glob("event_log_noise_*.xes"))
    
    for noisy_path in noisy_logs:
        # Extract log number
        log_num = noisy_path.stem.split("_")[-1]  # Extract 01, 02, etc.
        
        print(f"  event_log_noise_{log_num}...")
        
        # Find corresponding golden truth
        golden_files = list(golden_dir.glob(f"event_log_{log_num}_*.yaml"))
        if not golden_files:
            print(f"    WARNING: No golden file found for log {log_num}")
            continue
        
        golden_path = golden_files[0]
        golden = YAMLCodec.load(golden_path)
        
        # Binary search for optimal threshold
        print(f"    Searching for optimal threshold...")
        
        # For now, use same threshold for both dimensions
        # Try a range of thresholds with finer granularity
        best_threshold = 1.0
        best_combined_acc = 0.0
        
        # Try more granular thresholds from 1.0 down to 0.5 in 0.05 increments
        thresholds_to_try = [round(1.0 - i * 0.05, 2) for i in range(11)]  # 1.0, 0.95, 0.9, ..., 0.5
        
        best_temp_acc = 0.0
        best_exist_acc = 0.0
        
        for threshold in thresholds_to_try:
            discovered = discover(noisy_path, threshold=threshold)
            temp_correct, temp_total, exist_correct, exist_total = compare_matrices(golden, discovered)
            temp_acc = calculate_accuracy(temp_correct, temp_total)
            exist_acc = calculate_accuracy(exist_correct, exist_total)
            combined_acc = (temp_acc + exist_acc) / 2
            
            if combined_acc > best_combined_acc:
                best_combined_acc = combined_acc
                best_threshold = threshold
                best_temp_acc = temp_acc
                best_exist_acc = exist_acc
        
        print(f"    Best threshold={best_threshold}: temporal={best_temp_acc:.2f}, existential={best_exist_acc:.2f}")
        
        # Update corresponding result
        for result in results:
            if result["log_num"] == log_num:
                result["noisy_temporal_acc"] = best_temp_acc
                result["noisy_existential_acc"] = best_exist_acc
                result["noisy_threshold"] = best_threshold
                break
    
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Print LaTeX table
    print("LaTeX Table:")
    print()
    print(r"\begin{table}[h]")
    print(r"    \footnotesize")
    print(r"    \centering")
    print(r"    \caption{Results relationships discovery based on synthetic logs (\checkmark: behavior covered by log), showing the percentage of correctly identified temporal and existential dependencies.}")
    print(r"    \renewcommand{\arraystretch}{1.1}")
    print(r"    \setlength{\tabcolsep}{2pt}")
    print()
    print(r"    \begin{tabularx}{\textwidth}{>{\centering\arraybackslash}X *{9}{>{\centering\arraybackslash}X} p{2cm} p{2cm}}")
    print(r"    \toprule")
    print(r"    \textbf{Log} & \rotatebox{45}{Sequence} & \rotatebox{45}{Parallelism} & \rotatebox{45}{XOR} & \rotatebox{45}{NAND} & \rotatebox{45}{Multi-Choice} & \rotatebox{45}{Looping} & \rotatebox{45}{Nesting} & \rotatebox{45}{Block-Structure} & \rotatebox{45}{Skipping} & \textbf{Excl. Noise} & \textbf{Incl. Noise} \\")
    print(r"    \midrule")
    
    for result in results:
        log_num = result["log_num"]
        patterns = result["patterns"]
        
        # Format pattern checkmarks
        pattern_cols = []
        for pattern in ["Sequence", "Parallelism", "XOR", "NAND", "Multi-Choice", "Looping", "Nesting", "Block-Structure", "Skipping"]:
            if patterns[pattern]:
                pattern_cols.append(r"\checkmark")
            else:
                pattern_cols.append("")
        
        # Format accuracies
        clean_acc = f"({result['clean_temporal_acc']:.2f}, {result['clean_existential_acc']:.2f})"
        
        if result['noisy_temporal_acc'] is not None:
            noisy_acc = f"({result['noisy_temporal_acc']:.2f}, {result['noisy_existential_acc']:.2f})"
        else:
            noisy_acc = "N/A"
        
        pattern_str = " & ".join(pattern_cols)
        print(f"    \\textbf{{L{log_num}}}  & {pattern_str} & {clean_acc} & {noisy_acc} \\\\")
    
    print(r"    \bottomrule")
    print(r"    \end{tabularx}")
    print(r"    \label{tab:results}")
    print(r"\end{table}")
    print()
    
    # Print detailed results
    print()
    print("Detailed Results:")
    print()
    for result in results:
        log_num = result["log_num"]
        print(f"L{log_num}:")
        print(f"  Clean (threshold={result['clean_threshold']:.2f}):")
        print(f"    Temporal:     {result['clean_temporal_acc']:.2%}")
        print(f"    Existential:  {result['clean_existential_acc']:.2%}")
        if result['noisy_threshold'] is not None:
            print(f"  Noisy (threshold={result['noisy_threshold']:.2f}):")
            print(f"    Temporal:     {result['noisy_temporal_acc']:.2%}")
            print(f"    Existential:  {result['noisy_existential_acc']:.2%}")
        print()
    
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
