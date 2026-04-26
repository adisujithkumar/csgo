#!/usr/bin/env python3
"""Analyze real CS2 demo files for map control.

This script loads real professional demo files, validates rounds to filter
out knife rounds, and runs map control analysis on valid competitive rounds.
"""

import json
import time
from pathlib import Path
from typing import Tuple

import polars as pl
import awpy.demo
import awpy.data
from awpy.stats import map_control as mc


def check_nav_mesh_exists(map_name: str) -> bool:
    """Check if nav mesh exists for map.

    Args:
        map_name: Name of the map (e.g., 'de_dust2')

    Returns:
        True if nav mesh file exists
    """
    nav_path = awpy.data.NAVS_DIR / f"{map_name}.json"
    return nav_path.exists()


def is_valid_competitive_round(demo: awpy.demo.Demo, round_num: int) -> Tuple[bool, str]:
    """Check if a round is a valid competitive round (not knife, warmup, etc.).

    awpy already filters warmup/timeout via demo.in_play_ticks, but knife
    rounds might still exist as "valid" rounds.

    Args:
        demo: Parsed demo object
        round_num: Round number to check

    Returns:
        Tuple of (is_valid, reason_string)
    """
    # Get round data
    round_data = demo.ticks.filter(pl.col("round_num") == round_num)

    if len(round_data) == 0:
        return False, "No tick data"

    # Check 1: Round should have enough ticks (knife rounds are usually very short)
    num_ticks = len(set(round_data['tick']))
    if num_ticks < 50:  # Knife rounds typically < 50 ticks
        return False, f"Too short ({num_ticks} ticks) - likely knife round"

    # Check 2: Round should have reasonable duration (not instant win/loss)
    round_info = demo.rounds.filter(pl.col("round_num") == round_num)
    if len(round_info) > 0:
        start_tick = round_info['start'][0]
        end_tick = round_info['end'][0]
        duration_ticks = end_tick - start_tick

        # Normal rounds: 1900 ticks (~115 seconds at 64 tick)
        # Knife rounds: usually < 500 ticks (~30 seconds)
        if duration_ticks < 500:
            return False, f"Too short duration ({duration_ticks} ticks)"

    return True, "Valid competitive round"


def load_and_validate_demo(demo_path: Path) -> Tuple[awpy.demo.Demo, dict]:
    """Load demo and check basic validity.

    Note: awpy already filters out warmup and timeouts via demo.in_play_ticks.
    demo.ticks only contains valid in-play ticks where:
      - is_match_started == True
      - is_warmup_period == False
      - is_*_timeout == False

    Args:
        demo_path: Path to demo file

    Returns:
        Tuple of (demo object, info dict)
    """
    demo = awpy.demo.Demo(path=str(demo_path))
    demo.parse()

    info = {
        'map_name': demo.header['map_name'],
        'num_rounds': len(demo.rounds),
        'num_ticks': len(demo.ticks),
        'has_nav_mesh': check_nav_mesh_exists(demo.header['map_name'])
    }
    return demo, info


def analyze_round(demo: awpy.demo.Demo, round_num: int) -> dict:
    """Analyze single round for map control.

    Args:
        demo: Parsed demo object
        round_num: Round number to analyze

    Returns:
        Dictionary with analysis results
    """
    round_data = demo.ticks.filter(pl.col("round_num") == round_num)

    start_time = time.perf_counter()

    try:
        metrics = mc.get_round_map_control_frame_metrics(
            demo.header['map_name'],
            round_data
        )

        elapsed = time.perf_counter() - start_time

        return {
            'round_num': round_num,
            'success': True,
            'num_ticks': len(set(round_data['tick'])),
            'num_metrics': len(metrics),
            'processing_time': round(elapsed, 3),
            'metrics': metrics,
            'stats': {
                'min': float(min(metrics)),
                'max': float(max(metrics)),
                'mean': float(sum(metrics) / len(metrics))
            }
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time

        return {
            'round_num': round_num,
            'success': False,
            'error': str(e),
            'processing_time': round(elapsed, 3)
        }


def main():
    """Analyze all extracted de_dust2 demos."""
    demo_dir = Path("test_data/real_demos/de_dust2")
    results_dir = Path("test_data/results")

    print("=" * 70)
    print("CS2 Demo Analysis Script - Map Control Testing")
    print("=" * 70)
    print()

    demo_files = list(demo_dir.glob("*.dem"))

    if not demo_files:
        print("⚠️  No demo files found in", demo_dir)
        return

    print(f"Found {len(demo_files)} demo(s) to analyze")
    print()

    for demo_idx, demo_path in enumerate(demo_files, 1):
        print("=" * 70)
        print(f"Demo {demo_idx}/{len(demo_files)}: {demo_path.name}")
        print("=" * 70)

        # Load demo
        print("\nStep 1: Loading and parsing demo...")
        try:
            demo, info = load_and_validate_demo(demo_path)
        except Exception as e:
            print(f"  ✗ Failed to load demo: {e}")
            continue

        print(f"  ✓ Loaded successfully")
        print(f"  Map: {info['map_name']}")
        print(f"  Rounds: {info['num_rounds']}")
        print(f"  Total ticks: {info['num_ticks']:,}")
        print(f"  Nav mesh: {'✓ Available' if info['has_nav_mesh'] else '✗ Not available'}")
        print(f"  Note: awpy auto-filters warmup/timeouts from ticks")

        if not info['has_nav_mesh']:
            print(f"\n  ⚠️  Skipping analysis - no nav mesh for {info['map_name']}")
            continue

        # Find valid competitive rounds (skip knife rounds)
        print(f"\nStep 2: Validating rounds (checking first 10)...")
        valid_rounds = []

        for round_num in range(1, min(11, info['num_rounds'] + 1)):
            is_valid, reason = is_valid_competitive_round(demo, round_num)
            if is_valid:
                valid_rounds.append(round_num)
                if len(valid_rounds) <= 5:  # Print first 5
                    print(f"  ✓ Round {round_num}: {reason}")
            else:
                print(f"  ⚠️  Round {round_num}: {reason} - SKIPPING")

        if not valid_rounds:
            print("\n  ✗ No valid competitive rounds found!")
            continue

        print(f"\n  ✓ Found {len(valid_rounds)} valid competitive round(s)")

        # Analyze first 5 valid competitive rounds
        print(f"\nStep 3: Running map control analysis (first 5 valid rounds)...")
        round_results = []

        for round_num in valid_rounds[:5]:
            print(f"\n  Analyzing round {round_num}...")
            result = analyze_round(demo, round_num)
            round_results.append(result)

            if result['success']:
                print(f"    ✓ Success!")
                print(f"      Ticks: {result['num_ticks']}")
                print(f"      Metrics: {result['num_metrics']}")
                print(f"      Range: [{result['stats']['min']:.3f}, {result['stats']['max']:.3f}]")
                print(f"      Mean: {result['stats']['mean']:.3f}")
                print(f"      Processing time: {result['processing_time']}s")
            else:
                print(f"    ✗ Failed: {result['error']}")
                print(f"      Processing time: {result['processing_time']}s")

        # Save results
        print(f"\nStep 4: Saving results...")
        output = {
            'demo_path': str(demo_path),
            'demo_name': demo_path.name,
            'info': info,
            'valid_rounds': valid_rounds,
            'rounds_analyzed': len(round_results),
            'rounds': round_results
        }

        results_file = results_dir / "metrics" / f"{demo_path.stem}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  ✓ Results saved to: {results_file.relative_to(Path.cwd())}")

    print()
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run 'python3 scripts/validate_results.py' to validate metrics")
    print("  2. Run 'python3 scripts/generate_visualizations.py' to create plots")


if __name__ == "__main__":
    main()
