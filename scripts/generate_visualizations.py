#!/usr/bin/env python3
"""Generate sample visualizations from map control analysis results.

This script reads analysis results and generates:
1. Frame heatmaps showing tile-level control at specific moments
2. Timeline graphs showing metric progression over entire rounds
"""

from pathlib import Path

import polars as pl
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import awpy.demo
from awpy.stats import map_control as mc
from awpy.plot import map_control as mc_plot


def generate_frame_visualization(
    demo: awpy.demo.Demo,
    round_num: int,
    tick_idx: int,
    output_path: Path
):
    """Generate single frame heatmap.

    Args:
        demo: Parsed demo object
        round_num: Round number to visualize
        tick_idx: Index of tick within round (None = use middle tick)
        output_path: Path to save visualization
    """
    round_data = demo.ticks.filter(pl.col("round_num") == round_num)
    unique_ticks = sorted(set(round_data['tick']))

    if tick_idx is None or tick_idx >= len(unique_ticks):
        tick_idx = len(unique_ticks) // 2  # Use middle tick

    tick_num = unique_ticks[tick_idx]
    tick_data = round_data.filter(pl.col("tick") == tick_num)

    # Get tile metrics
    tile_values = mc.get_tick_map_control_tile_values(
        demo.header['map_name'],
        tick_data
    )
    tile_metrics = mc._generate_FrameMapControlTileMetrics_object(tile_values)

    # Plot
    fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
        tile_metrics,
        demo.header['map_name']
    )
    ax.set_title(f"Map Control - Round {round_num}, Tick {tick_num}", fontsize=14)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved frame: {output_path.name}")


def generate_timeline_visualization(
    demo: awpy.demo.Demo,
    round_num: int,
    output_path: Path
):
    """Generate timeline graph.

    Args:
        demo: Parsed demo object
        round_num: Round number to visualize
        output_path: Path to save visualization
    """
    round_data = demo.ticks.filter(pl.col("round_num") == round_num)

    metrics = mc.get_round_map_control_frame_metrics(
        demo.header['map_name'],
        round_data
    )

    fig, ax = mc_plot.plot_round_map_control_frame_metrics(
        metrics,
        demo.header['map_name']
    )
    ax.set_title(f"Map Control Timeline - Round {round_num}", fontsize=14)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved timeline: {output_path.name}")


def main():
    """Generate sample visualizations for first demo."""
    demo_dir = Path("test_data/real_demos/de_dust2")
    vis_dir = Path("test_data/results/visualizations")

    print("=" * 70)
    print("Map Control Visualization Generation")
    print("=" * 70)
    print()

    # Ensure output directory exists
    vis_dir.mkdir(parents=True, exist_ok=True)

    demo_files = list(demo_dir.glob("*.dem"))
    if not demo_files:
        print("⚠️  No demos found")
        return

    # Use first demo only
    demo_path = demo_files[0]
    print(f"Demo: {demo_path.name}")
    print()

    print("Step 1: Loading and parsing demo...")
    demo = awpy.demo.Demo(path=str(demo_path))
    demo.parse()
    print(f"  ✓ Loaded successfully")
    print(f"  Map: {demo.header['map_name']}")
    print(f"  Rounds: {len(demo.rounds)}")
    print()

    # Generate visualizations for rounds 1 and 2
    print("Step 2: Generating frame heatmaps...")
    try:
        for round_num in [1, 2]:
            output_path = vis_dir / f"{demo_path.stem}_round{round_num}_frame.png"
            generate_frame_visualization(demo, round_num, tick_idx=None, output_path=output_path)
    except Exception as e:
        print(f"  ✗ Error generating frame visualizations: {e}")

    print()
    print("Step 3: Generating timeline graphs...")
    try:
        for round_num in [1, 2]:
            output_path = vis_dir / f"{demo_path.stem}_round{round_num}_timeline.png"
            generate_timeline_visualization(demo, round_num, output_path=output_path)
    except Exception as e:
        print(f"  ✗ Error generating timeline visualizations: {e}")

    print()
    print("=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"Output directory: {vis_dir.absolute()}")
    print()

    # List generated files
    vis_files = list(vis_dir.glob("*.png"))
    if vis_files:
        print("Generated visualizations:")
        for vis_file in sorted(vis_files):
            size_kb = vis_file.stat().st_size / 1024
            print(f"  - {vis_file.name} ({size_kb:.1f} KB)")
    else:
        print("⚠️  No visualizations were generated")

    print()


if __name__ == "__main__":
    main()
