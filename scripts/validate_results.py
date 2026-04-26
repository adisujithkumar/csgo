#!/usr/bin/env python3
"""Validate map control analysis results.

This script reads the JSON results from demo analysis and validates that:
1. All metrics are in the valid range [0, 1]
2. No NaN or Inf values exist
3. Statistical properties make sense (mean around 0.5 for balanced matches)
"""

import json
import math
import statistics
from pathlib import Path


def validate_metrics(metrics: list[float]) -> dict:
    """Validate metric correctness.

    Args:
        metrics: List of map control metrics (should be in [0, 1] range)

    Returns:
        Dictionary with validation results
    """
    issues = []

    # Check range
    for i, m in enumerate(metrics):
        if not (0.0 <= m <= 1.0):
            issues.append(f"Tick {i}: metric {m:.3f} out of range [0, 1]")

    # Check for invalid values
    nan_count = sum(1 for m in metrics if math.isnan(m))
    inf_count = sum(1 for m in metrics if math.isinf(m))

    if nan_count > 0:
        issues.append(f"Contains {nan_count} NaN values")
    if inf_count > 0:
        issues.append(f"Contains {inf_count} Inf values")

    # Statistical checks
    if metrics:
        mean_metric = sum(metrics) / len(metrics)
        std_metric = statistics.stdev(metrics) if len(metrics) > 1 else 0.0

        # Mean should be somewhere between pure T (0) and pure CT (1)
        # For balanced matches, expect roughly 0.3-0.7 range
        if not (0.1 <= mean_metric <= 0.9):
            issues.append(
                f"Mean metric {mean_metric:.3f} suspicious "
                f"(expected 0.1-0.9 for valid match)"
            )

        stats = {
            'min': float(min(metrics)),
            'max': float(max(metrics)),
            'mean': float(mean_metric),
            'std': float(std_metric),
            'num_metrics': len(metrics)
        }
    else:
        stats = {
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'num_metrics': 0
        }

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'stats': stats
    }


def main():
    """Validate all saved results."""
    results_dir = Path("test_data/results/metrics")

    print("=" * 70)
    print("Map Control Results Validation")
    print("=" * 70)
    print()

    if not results_dir.exists():
        print(f"⚠️  Results directory not found: {results_dir}")
        return

    results_files = list(results_dir.glob("*.json"))

    if not results_files:
        print("⚠️  No results files found")
        return

    print(f"Found {len(results_files)} result file(s) to validate")
    print()

    total_rounds = 0
    valid_rounds = 0
    failed_rounds = 0
    validation_issues = 0

    for results_file in results_files:
        with open(results_file) as f:
            data = json.load(f)

        print("=" * 70)
        print(f"File: {results_file.name}")
        print("=" * 70)
        print(f"Demo: {data['demo_name']}")
        print(f"Map: {data['info']['map_name']}")
        print(f"Total rounds in demo: {data['info']['num_rounds']}")
        print(f"Valid competitive rounds: {len(data.get('valid_rounds', []))}")
        print(f"Rounds analyzed: {data.get('rounds_analyzed', 0)}")
        print()

        for round_result in data['rounds']:
            total_rounds += 1

            print(f"Round {round_result['round_num']}:")

            if not round_result['success']:
                print(f"  ✗ Analysis failed: {round_result['error'][:100]}...")
                failed_rounds += 1
                continue

            # Validate metrics
            validation = validate_metrics(round_result['metrics'])

            if validation['valid']:
                print(f"  ✓ Valid - All checks passed")
                print(f"    Min: {validation['stats']['min']:.3f}")
                print(f"    Max: {validation['stats']['max']:.3f}")
                print(f"    Mean: {validation['stats']['mean']:.3f}")
                print(f"    Std: {validation['stats']['std']:.3f}")
                print(f"    Ticks: {validation['stats']['num_metrics']}")
                valid_rounds += 1
            else:
                print(f"  ✗ Validation issues found:")
                validation_issues += len(validation['issues'])
                for issue in validation['issues']:
                    print(f"    - {issue}")

            print()

        print()

    # Summary
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print(f"Total rounds checked: {total_rounds}")
    print(f"Valid rounds: {valid_rounds}")
    print(f"Failed analysis: {failed_rounds}")
    print(f"Validation issues: {validation_issues}")
    print()

    if validation_issues == 0 and failed_rounds == 0:
        print("✓ All rounds passed validation!")
    elif validation_issues == 0 and failed_rounds > 0:
        print(f"⚠️  {failed_rounds} round(s) failed analysis (see errors above)")
    else:
        print(f"✗ Found {validation_issues} validation issue(s)")

    print()


if __name__ == "__main__":
    main()
