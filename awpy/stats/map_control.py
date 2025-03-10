from dataclasses import dataclass
from typing import TypeAlias

# Type to represent tile id for navigation tiles.
TileId: TypeAlias = int

# Contains map control values for one team.
# Maps TileId to list of tile map control values.
TeamMapControlValues: TypeAlias = dict[TileId, list[float]]

@dataclass
class FrameMapControlValues:
    """Dataclass with map control values for both teams in frame.

    Holds TeamMapControlValues for each team for a certain frame.
    """

    t_values: TeamMapControlValues
    ct_values: TeamMapControlValues

### Core Public Facing Functions
def get_frame_map_control_values(map_name: str, frame: Frame) -> FrameMapControlValues:
    """Get FrameMapControlValues object for a given frame."""
    # Get player positions for all alive players for each side

    # Get map control values (TeamMapControlValues) for each side
    # Call _positions_to_map_control_values() for each side

    # Return FrameMapControlValues object with above data
    return NotImplementedError("")

def get_frame_map_control_metric(map_name: str, frame: Frame) -> float:
    """Get Map Control Metric for given frame."""
    # Get frame map control values for tiles
    # Call get_frame_map_control_metric()

    # Do map control algorithm on FrameMapControlValues object (see above)
    # Call _calculate_map_control_metric()

    # Return FrameMapControlValues object with above data

    return NotImplementedError("")


### Helper Functions
def _positions_to_map_control_values(player_positions: list[tuple[float]], map_name: str) -> TeamMapControlValues:
    """Given list of player positions return TeamMapControlValues object."""
    # Convert each passed in player position into NavArea area_id

    # Get map control values for each player's current tile	
    # call _bfs()

    # Return TeamMapControlValues object with above data
    return NotImplementedError("")

def _bfs(source_tiles: list[TileId], map_name: str) -> TeamMapControlValues:
    """Given list of TileIds return TeamMapControlValues object."""
    # Initialize TeamMapControlValues return object (defaultdict with list)

    # Iterate through each source_tile provided and perform bfs with depth 10

        # Each tile gets the value (10 - steps)/10 appended to its list

    # Return TeamMapControlValues object
    return NotImplementedError("")

def _calculate_frame_map_control_metric(map_control_values: TeamMapControlValues, map_name:str) -> float:
    """Calculate map control metric for frame given TeamMapControlValues object."""
    # Iterate through all tiles (both T and CT) with map control values and get a list of map control metrics for each tile
    # # For each tile, a map control metric is calculated with the below equation:
    # # sum(CT_values_for_tile) / (sum(CT_values_for_tile) + sum(T_values_for_tile))

    # Calculate a weighted average for the list of map control metrics where the weights is the area of the tiles

    # Return weighted average as frame map control metric
    return NotImplementedError("")
