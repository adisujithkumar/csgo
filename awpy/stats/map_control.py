from dataclasses import dataclass
from typing import TypeAlias

# Type to represent tile id for navigation tiles.
TileId: TypeAlias = int

# Type to represent map control values for a singular tile
# Created from running bfs from a list of tiles. One tile can have multiple MC values.
# See _generate_FrameMapControlTileMetrics_object for context
MapControlTileValue: TypeAlias = float

# Type to represent map control metric for a singular tile
# Created from ratio of CT MC Values to total MC values for given tile.
# Float ranging between 0 [T] and 1 [CT] to represent who controls the tile
# See _generate_FrameMapControlTileMetrics_object for context
MapControlTileMetric: TypeAlias = float

# Type to represent map control metric for an entire frame
# Created from weighted average of values in FrameMapControlTileMetrics object
# See _calculate_frame_map_control_metric for context
MapControlFrameMetric: TypeAlias = float

# Contains map control tile values for one team.
# Maps TileId to list of MapControlTileValue.
TeamMapControlTileValues: TypeAlias = dict[TileId, list[MapControlTileValue]]

# Contains map control tile metrics for each relevant tile
# Maps TileId to MapControlTileMetric
# See _generate_FrameMapControlTileMetrics_object for context
FrameMapControlTileMetrics: TypeAlias = dict[TileId, MapControlTileMetric]

@dataclass
class FrameMapControlTileValues:
    """Dataclass with map control values for both teams in frame.

    Holds TeamMapControlTileValues for each team for a certain frame.
    """

    t_values: TeamMapControlTileValues
    ct_values: TeamMapControlTileValues

### Core Functions
def get_frame_map_control_tile_values(map_name: str, frame: Frame) -> FrameMapControlTileValues:
    """Get FrameMapControlTileValues object for a given frame."""
    # Get player positions for all alive players for each side

    # Get map control tile values (TeamMapControlTileValues) for each side
    # Call _positions_to_map_control_values() for each side

    # Return FrameMapControlTileValues object with above data
    return NotImplementedError("")

def get_map_control_frame_metric(map_name: str, frame: Frame) -> MapControlFrameMetric:
    """Get MapControlFrameMetric for given frame."""
    # Get FrameMapControlTileValues objects for tiles
    # Call get_frame_map_control_tile_values()

    # Get MapControlFrameMetric from FrameMapControlTileValues object
    # Call _calculate_map_control_frame_metric()

    # Return MapControlFrameMetric
    return NotImplementedError("")

def get_round_map_control_frame_metrics(map_name: str, _round: Round) -> list[MapControlFrameMetric]:
    """Get MapControlFrameMetric objects for each frame in a given round."""
    # Initialize an empty list

    # Get FrameMapControlTileMetrics object from FrameMapControlTileValues object (see _generate_FrameMapControlTileMetrics_object for more context)

    # Iterate through each frame in the given round
    # For each frame, calculate the FrameMapControlMetric for that frame with get_map_control_frame_metric and add it to the list

    # Return list of floats (where the float at index i is the FrameMapControlMetric for frame i in the given round)
    return NotImplementedError("")

### Helper Functions
def _positions_to_map_control_values(player_positions: list[tuple[float]], map_name: str) -> TeamMapControlTileValues:
    """Given list of player positions return TeamMapControlTileValues object."""
    # Convert each passed in player position into NavArea area_id

    # Get map control values for each player's current tile
    # call _bfs()

    # Return TeamMapControlTileValues object with above data
    return NotImplementedError("")

def _bfs(source_tiles: list[TileId], map_name: str) -> TeamMapControlTileValues:
    """Given list of TileIds return TeamMapControlTileValues object."""
    # Initialize TeamMapControlTileValues return object (defaultdict with list)

    # Iterate through each source_tile provided and perform bfs with depth 10

        # Each tile gets the value (10 - steps)/10 appended to its list in TeamMapControlTileValues object when the tile is reached.
        # This above value is an example of MapControlTileValue.

    # Return TeamMapControlTileValues object
    return NotImplementedError("")

def _calculate_map_control_frame_metric(map_control_values: FrameMapControlTileValues, map_name:str) -> MapControlFrameMetric:
    """Calculate MapControlFrameMetrics for given FrameMapControlTileValues object."""
    # Get a Map Control Tile Metric for each relevant tile
    # Call _generate_FrameMapControlTileMetrics_object()

    # Calculate a weighted average for the list of map control metrics where the weights is the area of the tiles

    # Return weighted average as frame map control metric
    return NotImplementedError("")

def _generate_FrameMapControlTileMetrics_object(map_control_values: FrameMapControlTileValues) -> FrameMapControlTileMetrics:
    """Generate FrameMapControlTileMetrics object from FrameMapControlTileValues object.

    Iterate through all tiles (both T and CT) with map control tiles values and
    create a map mapping from TileId to MapControlTileMetric

    For each tile, a map control metric is calculated with the below equation:
    sum(CT_values_for_tile) / (sum(CT_values_for_tile) + sum(T_values_for_tile))
    """
    # Initialize defaultdict(float) object

    # Iterate through all tiles and calculate map control metric with above equation
    # Save map control metric for TileId in FrameMapControlTileMetrics object

    # Return FrameMapControlTileMetrics object
    return NotImplementedError("")
