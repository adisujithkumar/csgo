from awpy.stats.map_control import FrameMapControlTileMetrics, MapControlFrameMetric

def plot_frame_map_control_tile_metrics(map_control_metrics: FrameMapControlTileMetrics, map_name: str) -> figure, Axes:
    """For a given FrameMapControlTileMetrics object, plot a visualization of the map control metrics."""
    # Plot the minimap png for the map given

    # Desiginate a T and CT color (RGB value) - Default can be R and G

    # Iterate through every tile in map_control_metrics giving a RGB color value to each tile.
    # The tile's color will be weighted average between T and CT colors where the weight for
    # T is 1 - the tile's MC Value and the weight for the CT is the tile's MC Value.
    # Tile_Color = T_Color * (1 - Tile_MC_Metric) + CT_Color * (Tile_MC_Metric).

    # Color each tile with the color outputted from the above equation
    # Could leverage plot.tile here

    # Optionally plot player positions as a simple dot if wanted

    # Optionally save the plot to image if wanted

    # Return the figure, Axes objects as a tuple

    return NotImplementedError("")

def save_round_map_control_tile_metrics_gif(round_metrics: list[FrameMapControlTileMetrics], map_name: str) -> None:
    """For a given list of FrameMapControlTileMetrics, save a gif to file visualizing the tiles' map control metrics over time."""
    # Create a temporary folder to save each frame of the gif

    # Iterate through round_metrics
    # For each FrameMapControlTileMetrics object, call plot_frame_tile_map_control_metrics and
    # save the visualization to the temporary folder.

    # Generate gif from temporary folder

    # Save gif to file

    # Delete temporary folder

    return NotImplementedError("")

def plot_round_map_control_frame_metrics(round_metrics: list[MapControlFrameMetric], map_name: str) -> figure, Axes:
    """For an entire round's list of MapControlFrameMetric objects, plot them to a graph"""
    # Plot given list of MapControlFrameMetric objects (floats) as a line graph

    # Return figure, Axes tuple
    return NotImplementedError("")
