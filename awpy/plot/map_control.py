"""Module for plotting map control visualizations."""

import io
import math

import matplotlib.pyplot as plt
import polars as pl
import matplotlib.image as mpimg
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from loguru import logger
import tqdm

import awpy.data
import awpy.plot.utils
import awpy.vector
from awpy.stats.map_control import (
    FrameMapControlTileMetrics,
    MapControlFrameMetric,
    _get_nav,
)

# Team colors - Red for T, Green for CT
T_COLOR = (1.0, 0.0, 0.0)  # Red
CT_COLOR = (0.0, 1.0, 0.0)  # Green
CT_PLAYER_COLOR = (0.0, 0.8, 1.0)  # Cyan - distinct from CT_COLOR green

# Custom Red-Yellow-Green colormap for better visual differentiation
# 0.0 (T control) = Red, 0.5 (Neutral) = Yellow, 1.0 (CT control) = Green
_MAP_CONTROL_CMAP = LinearSegmentedColormap.from_list(
    'map_control',
    ['#FF0000',  # Pure red at 0.0 (T control)
     '#FF4500',  # Red-orange at ~0.17
     '#FFA500',  # Orange at ~0.33
     '#FFFF00',  # Yellow at 0.5 (neutral)
     '#ADFF2F',  # Yellow-green at ~0.67
     '#7FFF00',  # Chartreuse at ~0.83
     '#00FF00'], # Pure green at 1.0 (CT control)
    N=256
)


def _validate_tick_data(tick_data: pl.DataFrame) -> None:
    """Validate that tick data has required columns.

    Args:
        tick_data: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    required = ["X", "Y", "Z", "yaw", "pitch", "health", "side", "name"]
    missing = [col for col in required if col not in tick_data.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)


def _extract_alive_players(tick_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extract alive T and CT players from tick data.

    Filters players by health > 0 and separates by team.

    Args:
        tick_data: Polars DataFrame containing player data for a single tick.

    Returns:
        Tuple of (t_players_df, ct_players_df) containing alive players for each team.
    """
    alive_players = tick_data.filter(pl.col("health") > 0)
    t_players = alive_players.filter(pl.col("side") == "t")
    ct_players = alive_players.filter(pl.col("side") == "ct")
    return t_players, ct_players


def _plot_players(
    axes: Axes,
    players_df: pl.DataFrame,
    map_name: str,
    team_color: tuple[float, float, float],
    marker_size: float,
    arrow_length_multiplier: float,
    show_names: bool
) -> None:
    """Plot player positions with directional arrows on the axes.

    Args:
        axes: Matplotlib axes to plot on.
        players_df: DataFrame containing player data (X, Y, Z, yaw, pitch, name).
        map_name: Name of the map for coordinate transformation.
        team_color: RGB tuple for team color.
        marker_size: Size of player markers.
        arrow_length_multiplier: Multiplier for arrow length.
        show_names: Whether to display player names.
    """
    for row in players_df.iter_rows(named=True):
        # Convert game coordinates to pixel coordinates
        pixel_pos = awpy.plot.utils.game_to_pixel(map_name, (row["X"], row["Y"], row["Z"]))
        pixel_x, pixel_y = pixel_pos[0], pixel_pos[1]

        # Draw player marker (two circles: black outline + colored fill)
        axes.plot(pixel_x, pixel_y, marker='o', color='black',
                 markersize=marker_size, alpha=1.0, zorder=10)
        axes.plot(pixel_x, pixel_y, marker='o', color=team_color,
                 markersize=marker_size * 0.9, alpha=1.0, zorder=11)

        # Draw directional arrow (based on yaw angle)
        yaw = row["yaw"]
        pitch = row.get("pitch", 0.0)

        # Calculate arrow direction vector (reuse pattern from plot.py)
        dx = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
        dy = math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))

        # Calculate arrow endpoint
        line_length = marker_size * arrow_length_multiplier
        arrow_x = pixel_x + dx * line_length
        arrow_y = pixel_y + dy * line_length

        # Draw arrow
        axes.plot([pixel_x, arrow_x], [pixel_y, arrow_y],
                 color='black', linewidth=1.5, zorder=12, solid_capstyle='round')

        # Draw player name (if enabled)
        if show_names:
            axes.annotate(
                row["name"],
                (pixel_x, pixel_y - marker_size * 2),
                color='white',
                fontsize=6,
                weight='bold',
                zorder=13,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                         alpha=0.6, edgecolor='none')
            )


def _tile_polygon(area_corners: list[awpy.vector.Vector3], map_name: str) -> list[tuple[float, float]]:
    """Convert Nav Area corners to pixel coordinates.

    Args:
        area_corners: List of Vector3 corners from NavArea.
        map_name: Name of the map.

    Returns:
        List of (x, y) pixel coordinate tuples.
    """
    return [
        awpy.plot.utils.game_to_pixel(map_name, (c.x, c.y, c.z))[0:2]
        for c in area_corners
    ]


def _plot_tile(
    axis: Axes,
    polygon: list[tuple[float, float]],
    edgecolor: str,
    facecolor: str | tuple[float, float, float],
    linewidth: float = 1,
    alpha: float = 0.6
) -> None:
    """Add a tile polygon to the axis.

    Args:
        axis: Matplotlib axes to plot on.
        polygon: List of (x, y) coordinates defining the polygon.
        edgecolor: Color of the polygon edge.
        facecolor: Color of the polygon face (can be string or RGB tuple).
        linewidth: Width of the polygon edge.
        alpha: Transparency of the polygon.
    """
    patch = patches.Polygon(
        polygon,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        zorder=5
    )
    axis.add_patch(patch)


def plot_frame_map_control_tile_metrics(
    map_control_metrics: FrameMapControlTileMetrics,
    map_name: str
) -> tuple[Figure, Axes]:
    """Plot a visualization of map control metrics for a frame.

    For a given FrameMapControlTileMetrics object, plot each tile with a color
    representing the map control. Red represents T control (0.0), green represents
    CT control (1.0), and intermediate values are blended.

    Args:
        map_control_metrics: Dictionary mapping tile IDs to map control metrics (0-1).
        map_name: Name of the map to plot.

    Returns:
        Tuple of (Figure, Axes) matplotlib objects.

    Raises:
        FileNotFoundError: If map image or nav file is not found.
    """
    # Load map image
    image = f"{map_name}.png"
    map_img_path = awpy.data.MAPS_DIR / image
    if not map_img_path.exists():
        map_img_path_err = f"Map image not found: {map_img_path}. Might need to call `awpy get maps`"
        raise FileNotFoundError(map_img_path_err)

    # Load nav data (cached)
    map_data = _get_nav(map_name)

    # Create figure
    map_bg = mpimg.imread(map_img_path)
    figure, axes = plt.subplots(figsize=(1024 / 300, 1024 / 300), dpi=300)
    axes.imshow(map_bg, zorder=0)
    axes.axis("off")

    # Plot each tile with blended color
    for tile_id, mc_metric in map_control_metrics.items():
        # Get tile corners
        area = map_data.areas[tile_id]
        polygon = _tile_polygon(area.corners, map_name)

        # Use colormap for richer color gradient
        # Colormap returns RGBA tuple (R, G, B, A), we take only RGB
        facecolor = _MAP_CONTROL_CMAP(mc_metric)[:3]

        # Plot tile
        _plot_tile(
            axis=axes,
            polygon=polygon,
            edgecolor="black",
            facecolor=facecolor,
            linewidth=0.5,
            alpha=0.6
        )

    # Add legend
    legend_elements = [
        Patch(facecolor=T_COLOR, edgecolor='black', label='T Control'),
        Patch(facecolor=CT_COLOR, edgecolor='black', label='CT Control')
    ]
    axes.legend(handles=legend_elements, loc='upper right', framealpha=0.8)

    figure.patch.set_facecolor("black")
    plt.tight_layout()

    return figure, axes


def plot_round_map_control_frame_metrics(
    round_metrics: list[MapControlFrameMetric],
    map_name: str
) -> tuple[Figure, Axes]:
    """Plot map control metrics over time for an entire round.

    Creates a line graph showing how map control changes throughout a round.
    Values range from 0.0 (complete T control) to 1.0 (complete CT control).

    Args:
        round_metrics: List of MapControlFrameMetric values (floats 0-1) for each frame.
        map_name: Name of the map (currently unused but kept for API consistency).

    Returns:
        Tuple of (Figure, Axes) matplotlib objects.
    """
    figure, axes = plt.subplots(figsize=(10, 6))

    # Create frame numbers for x-axis
    frames = list(range(len(round_metrics)))

    # Plot the metrics as a blue line
    axes.plot(frames, round_metrics, color='blue', linewidth=2, label='Map Control')

    # Add horizontal line at 0.5 (neutral)
    axes.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add shaded regions
    axes.axhspan(0.0, 0.5, facecolor='red', alpha=0.1)  # T control region
    axes.axhspan(0.5, 1.0, facecolor='green', alpha=0.1)  # CT control region

    # Set labels and title
    axes.set_xlabel('Frame', fontsize=12)
    axes.set_ylabel('Map Control Metric', fontsize=12)
    axes.set_title(f'Map Control Over Time - {map_name}', fontsize=14, fontweight='bold')

    # Set y-axis limits and custom labels
    axes.set_ylim(0.0, 1.0)
    axes.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes.set_yticklabels(['T (0.0)', '0.25', 'Neutral (0.5)', '0.75', 'CT (1.0)'])

    # Add grid
    axes.grid(True, alpha=0.3)

    # Add legend
    axes.legend(loc='upper right')

    plt.tight_layout()

    return figure, axes


def plot_frame_map_control_with_players(
    map_control_metrics: FrameMapControlTileMetrics,
    tick_data: pl.DataFrame,
    map_name: str,
    show_player_names: bool = True,
    player_marker_size: float = 8.0,
    arrow_length_multiplier: float = 2.0
) -> tuple[Figure, Axes]:
    """Plot map control visualization with player positions and directional arrows.

    Extends the basic map control visualization by adding alive players with:
    - Circular markers (red for T, cyan for CT)
    - Directional arrows showing viewing direction (yaw angle)
    - Optional player names

    Args:
        map_control_metrics: Dictionary mapping tile IDs to map control metrics (0-1).
        tick_data: Polars DataFrame containing player data for this tick.
                   Must have columns: X, Y, Z, yaw, pitch, health, side, name.
        map_name: Name of the map to plot.
        show_player_names: Whether to display player names above markers. Defaults to True.
        player_marker_size: Size of player markers in points. Defaults to 8.0.
        arrow_length_multiplier: Multiplier for arrow length. Defaults to 2.0.

    Returns:
        Tuple of (Figure, Axes) matplotlib objects.

    Raises:
        FileNotFoundError: If map image or nav file is not found.
        ValueError: If tick_data is empty or missing required columns.

    Example:
        >>> from awpy.demo import Demo
        >>> from awpy.stats.map_control import get_tick_map_control_tile_values, _generate_FrameMapControlTileMetrics_object
        >>>
        >>> demo = Demo("match.dem")
        >>> demo.parse()
        >>> tick_data = demo.ticks.filter(pl.col("tick") == 1000)
        >>>
        >>> tile_values = get_tick_map_control_tile_values("de_dust2", tick_data)
        >>> tile_metrics = _generate_FrameMapControlTileMetrics_object(tile_values)
        >>>
        >>> fig, ax = plot_frame_map_control_with_players(
        ...     tile_metrics, tick_data, "de_dust2"
        ... )
        >>> fig.savefig("frame_with_players.png")
    """
    # 1. Validate tick data
    if tick_data.is_empty():
        msg = "tick_data cannot be empty"
        raise ValueError(msg)
    _validate_tick_data(tick_data)

    # 2. Create base map control visualization
    fig, axes = plot_frame_map_control_tile_metrics(map_control_metrics, map_name)

    # 3. Extract alive players by team
    t_players, ct_players = _extract_alive_players(tick_data)

    # 4. Plot T players (red markers)
    if len(t_players) > 0:
        _plot_players(
            axes, t_players, map_name, T_COLOR,
            player_marker_size, arrow_length_multiplier, show_player_names
        )

    # 5. Plot CT players (cyan markers)
    if len(ct_players) > 0:
        _plot_players(
            axes, ct_players, map_name, CT_PLAYER_COLOR,
            player_marker_size, arrow_length_multiplier, show_player_names
        )

    return fig, axes


def save_round_map_control_tile_metrics_gif(
    round_metrics: list[FrameMapControlTileMetrics],
    map_name: str,
    output_path: str = "map_control.gif",
    duration: int = 100
) -> None:
    """Save a GIF visualizing map control tiles over time.

    For a given list of FrameMapControlTileMetrics, create and save a GIF
    showing how map control changes throughout the round.

    Args:
        round_metrics: List of FrameMapControlTileMetrics objects for each frame.
        map_name: Name of the map to plot.
        output_path: Path where the GIF will be saved. Defaults to "map_control.gif".
        duration: Duration of each frame in milliseconds. Defaults to 100ms.

    Raises:
        FileNotFoundError: If map image or nav file is not found.
    """
    logger.info(f"Creating map control GIF with {len(round_metrics)} frames...")

    # Store PIL images for each frame
    pil_images = []

    # Generate each frame
    for i, frame_metrics in enumerate(tqdm.tqdm(round_metrics, desc="Generating frames")):
        # Create matplotlib figure for this frame
        fig, ax = plot_frame_map_control_tile_metrics(frame_metrics, map_name)

        # Add frame number to the plot
        ax.text(
            0.02, 0.98, f'Frame {i}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        pil_image = Image.open(buf)
        pil_images.append(pil_image.copy())
        buf.close()

        # Close the matplotlib figure to free memory
        plt.close(fig)

    # Save as GIF
    logger.info(f"Saving GIF to {output_path}...")
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )

    logger.info(f"GIF saved successfully to {output_path}")


def save_round_map_control_with_players_gif(
    round_data: pl.DataFrame,
    map_name: str,
    output_path: str = "map_control_players.gif",
    duration: int = 500,
    tick_interval: int = 384,
    show_player_names: bool = True,
    player_marker_size: float = 8.0
) -> None:
    """Save a GIF visualizing map control with player positions over time.

    Samples the round data at regular tick intervals (default every 3 seconds = 384 ticks)
    and creates an animated GIF showing map control and player movements.

    Args:
        round_data: Polars DataFrame containing tick data for entire round.
                    Must have columns: tick, X, Y, Z, yaw, pitch, health, side, name.
        map_name: Name of the map to plot.
        output_path: Path where the GIF will be saved. Defaults to "map_control_players.gif".
        duration: Duration of each frame in milliseconds. Defaults to 500ms.
        tick_interval: Number of ticks between samples. Defaults to 384 (3 seconds at 128 tick/sec).
        show_player_names: Whether to display player names. Defaults to True.
        player_marker_size: Size of player markers. Defaults to 8.0.

    Raises:
        FileNotFoundError: If map image or nav file is not found.
        ValueError: If round_data is empty or missing required columns.

    Example:
        >>> from awpy.demo import Demo
        >>>
        >>> demo = Demo("match.dem")
        >>> demo.parse()
        >>> round_1 = demo.ticks.filter(pl.col("round_num") == 1)
        >>>
        >>> save_round_map_control_with_players_gif(
        ...     round_1,
        ...     "de_dust2",
        ...     output_path="round1_control.gif",
        ...     duration=500,
        ...     tick_interval=384
        ... )
    """
    # 1. Validate inputs
    if round_data.is_empty():
        msg = "round_data cannot be empty"
        raise ValueError(msg)
    if tick_interval <= 0:
        msg = f"tick_interval must be positive, got: {tick_interval}"
        raise ValueError(msg)
    _validate_tick_data(round_data)

    logger.info(f"Creating map control + players GIF...")

    # 2. Sample ticks at interval
    sampled_data = round_data.filter(pl.col("tick") % tick_interval == 0)

    # 3. Group by tick (maintains order)
    tick_groups = list(sampled_data.group_by("tick", maintain_order=True))

    if len(tick_groups) == 0:
        msg = f"No ticks found matching interval {tick_interval}. Check tick values in round_data."
        raise ValueError(msg)

    logger.info(f"Generating {len(tick_groups)} frames (sampled every {tick_interval} ticks)...")

    # 4. Store PIL images
    pil_images = []

    # 5. Generate each frame
    for i, (tick_tuple, tick_df) in enumerate(tqdm.tqdm(tick_groups, desc="Generating frames")):
        tick_num = tick_tuple[0]  # Extract tick number from tuple

        # Skip if no players in this tick
        if len(tick_df) == 0:
            logger.warning(f"No players in tick {tick_num}, skipping frame")
            continue

        # Calculate map control metrics for this tick
        from awpy.stats.map_control import get_tick_map_control_tile_values, _generate_FrameMapControlTileMetrics_object

        tile_values = get_tick_map_control_tile_values(map_name, tick_df)
        tile_metrics = _generate_FrameMapControlTileMetrics_object(tile_values)

        # Create figure with players
        fig, ax = plot_frame_map_control_with_players(
            tile_metrics,
            tick_df,
            map_name,
            show_player_names=show_player_names,
            player_marker_size=player_marker_size
        )

        # Add frame/tick info
        ax.text(
            0.02, 0.98,
            f'Frame {i} | Tick {tick_num}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            color='white',
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )

        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        pil_image = Image.open(buf)
        pil_images.append(pil_image.copy())
        buf.close()

        # Close figure to free memory
        plt.close(fig)

    # 6. Save GIF
    if len(pil_images) == 0:
        msg = "No frames generated. Check if players exist in sampled ticks."
        raise ValueError(msg)

    logger.info(f"Saving GIF to {output_path}...")
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )

    logger.info(f"GIF saved successfully with {len(pil_images)} frames")
