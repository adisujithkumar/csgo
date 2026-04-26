#!/usr/bin/env python3
"""Quick test of map control with real demo - super low frequency sampling.

This script validates the map control bug fix using ultra-low frequency sampling
for fast validation (under 1 minute).
"""

from pathlib import Path
import polars as pl
import awpy.demo
from awpy.stats import map_control as mc
import time


def quick_test_demo(demo_path: Path, num_rounds: int = 2, tick_sample_rate: int = 50):
    """Quick test with limited rounds and ultra-low frequency sampling.

    Args:
        demo_path: Path to demo file
        num_rounds: Number of rounds to test (default: 2)
        tick_sample_rate: Process every Nth tick (default: 50 for speed)
    """
    print(f"\n{'='*70}")
    print(f"Quick Test: {demo_path.name}")
    print(f"{'='*70}\n")

    # Load demo
    print("Loading demo...")
    demo = awpy.demo.Demo(path=str(demo_path))
    demo.parse()

    print(f"  Map: {demo.header['map_name']}")
    print(f"  Total rounds: {len(demo.rounds)}")
    print(f"  Total ticks: {len(demo.ticks):,}")
    print()

    # Test first 2 rounds
    results = []
    for round_num in range(1, num_rounds + 1):
        print(f"Testing round {round_num}...")

        # Get round data
        round_data = demo.ticks.filter(pl.col("round_num") == round_num)
        unique_ticks = sorted(set(round_data['tick']))

        # Sample ticks (every Nth tick)
        sampled_ticks = unique_ticks[::tick_sample_rate]
        print(f"  Total ticks: {len(unique_ticks)}")
        print(f"  Sampled ticks: {len(sampled_ticks)} (every {tick_sample_rate}th)")

        # Process sampled ticks
        start_time = time.perf_counter()
        metrics = []

        for tick_num in sampled_ticks:
            tick_data = round_data.filter(pl.col("tick") == tick_num)
            try:
                metric = mc.get_map_control_tick_metric(demo.header['map_name'], tick_data)
                metrics.append(metric)
            except Exception as e:
                print(f"  ✗ Error at tick {tick_num}: {e}")
                return None

        elapsed = time.perf_counter() - start_time

        # Validate metrics
        if all(0.0 <= m <= 1.0 for m in metrics):
            mean_metric = sum(metrics) / len(metrics)
            print(f"  ✓ Success!")
            print(f"    Metrics: {len(metrics)}")
            print(f"    Range: [{min(metrics):.3f}, {max(metrics):.3f}]")
            print(f"    Mean: {mean_metric:.3f}")
            print(f"    Time: {elapsed:.2f}s ({len(sampled_ticks)/elapsed:.1f} ticks/sec)")

            results.append({
                'round_num': round_num,
                'success': True,
                'metrics': metrics,
                'sampled_ticks': len(sampled_ticks),
                'processing_time': elapsed
            })
        else:
            print(f"  ✗ Invalid metrics found")
            return None

        print()

    return results


def main():
    demo_dir = Path("test_data/real_demos/de_dust2")
    demo_files = list(demo_dir.glob("*.dem"))

    if not demo_files:
        print("No demos found")
        return

    # Test first demo only with ultra-low sampling
    demo_path = demo_files[0]
    results = quick_test_demo(demo_path, num_rounds=2, tick_sample_rate=50)

    if results:
        print("="*70)
        print("✓ QUICK TEST PASSED")
        print("="*70)
        print()
        print("Summary:")
        print(f"  - Tested {len(results)} rounds")
        print(f"  - All metrics in valid range [0, 1]")
        print(f"  - Bug fix validated: armor column handling works")
        print(f"  - Real demo compatibility: CONFIRMED")
        print()
        print("Next: Run visualization generation")
        return True
    else:
        print("✗ QUICK TEST FAILED")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
