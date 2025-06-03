from dataclasses import dataclass
from typing import TypeAlias
from collections import defaultdict

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
# TODO

def get_frame_map_control_tile_values(map_name: str, frame: Frame) -> FrameMapControlTileValues:
    """Get FrameMapControlTileValues object for a given frame."""
    # Get player positions for all alive players for each side

    # Get map control tile values (TeamMapControlTileValues) for each side
    # Call _positions_to_map_control_values() for each side

    # Return FrameMapControlTileValues object with above data
    return NotImplementedError("")

# TODO
def get_map_control_frame_metric(map_name: str, frame: Frame) -> MapControlFrameMetric:
    """Get MapControlFrameMetric for given frame."""
    # Get FrameMapControlTileValues objects for tiles
    # Call get_frame_map_control_tile_values()

    # Get MapControlFrameMetric from FrameMapControlTileValues object
    # Call _calculate_map_control_frame_metric()

    # Return MapControlFrameMetric
    return NotImplementedError("")

# TODO
def get_round_map_control_frame_metrics(map_name: str, _round: Round) -> list[MapControlFrameMetric]:
    """Get MapControlFrameMetric objects for each frame in a given round."""
    # Initialize an empty list

    # Get FrameMapControlTileMetrics object from FrameMapControlTileValues object (see _generate_FrameMapControlTileMetrics_object for more context)

    # Iterate through each frame in the given round
    # For each frame, calculate the FrameMapControlMetric for that frame with get_map_control_frame_metric and add it to the list

    # Return list of floats (where the float at index i is the FrameMapControlMetric for frame i in the given round)
    return NotImplementedError("")

### Helper Functions
# TODO
def _positions_to_map_control_values(player_positions: list[tuple[float]], map_name: str) -> TeamMapControlTileValues:
    """Given list of player positions return TeamMapControlTileValues object."""
    # Convert each passed in player position into NavArea area_id

    # Get map control values for each player's current tile
    # call _bfs()

    # Return TeamMapControlTileValues object with above data
    return NotImplementedError("")

def combine_two_TeamMapControlTileValues(dict1, dict2):
    combined = {}

    # Add entries from dict1
    for key, val in dict1.items():
        combined[key] = val.copy()
    
    # Merge with dict2
    for key, val in dict2.items():
        if key in combined:
            combined[key].extend(val)
        else:
            combined[key] = val.copy()
    return combined

def _bfs(source_tiles: list[TileId], map_name: str) -> TeamMapControlTileValues:
    """Given list of TileIds return TeamMapControlTileValues object."""
    # Initialize TeamMapControlTileValues return object (defaultdict with list)
    returnTeamMapControlTileValues = defaultdict(list)

    map_dataawpy.nav.Nav.from_json(path=f"../../tests/{map_name}.json")
    map_areas = map_data['areas']
    
    all_tile_mc_values = dict()
    for tile in source_tiles:
        tile_MC_values = _bfs_helper(tile)
        all_tile_mc_values = combine_two_TeamMapControlTileValues(all_tile_mc_values, tile_MC_values)
        
    return all_tile_mc_values

def _bfs_helper(source_tile: TileId) -> TeamMapControlTileValues:
    seen = set()
    returnPlayerMapControlTileValues = defaultdict(list)

    tileStack = [{"tileId": source_tile, "mcValue": 1, "lengthFromSource": 0}]

    while len(tileStack) > 0:
        currTile = tileStack.pop(0)
        currLength, nextlength = currTile["lengthFromSource"], currTile["lengthFromSource"] + 1
        currTileId = currTile['tileId']

        if currLength > 10:
            continue

        if not currTileId in seen:
            
        returnPlayerMapControlTileValues[currTile["tileId"]].append(currTile['mcValue'])
        seen.add(currTile["tileId"])

        currTileNeighbors = set(getNeighbors(currTile['tileId']))
        notSeenNeighbors = currTileNeighbors - seen
        for neighbor in notSeenNeighbors:
            tileStack.append({"tileId": neighbor.id, "mcValue": (10 - nextlength)/10, "lengthFromSource": nextlength})

    return returnPlayerMapControlTileValues
# TODO
def _get_polygon_area(xy: list[list[float]]):
    if len(coordinates) == 4:
        xy[0]p1]
         |(x₁y₂ + x₂y₃ + x₃y₄ + x₄y₁) - (x₂y₁ + x₃y₂ + x₄y₃ + x₁y₄)|

    elif len(coordinates) == 3:

# TODO
def _calculate_map_control_frame_metric(map_control_values: FrameMapControlTileValues, map_name:str) -> MapControlFrameMetric:
    """Calculate MapControlFrameMetrics for given FrameMapControlTileValues object."""
    # Get a Map Control Tile Metric for each relevant tile
    frame_mc_tile_metrics = _generate_FrameMapControlTileMetrics_object()

    # Calculate a weighted average for the list of map control metrics where the weights is the area of the tiles
    map_dataawpy.nav.Nav.from_json(path=f"../../tests/{map_name}.json")
    map_areas = map_data['areas']
    total_area = 0
    weighted_metric = 0
    for tile in frame_mc_tile_metrics.keys():
        tile_area = _get_polygon_area(map_areas.areas[tile].points)
        weighted_metric += tile_area * frame_mc_tile_metrics[tile]
        total_area += tile_area
    # Return weighted average as frame map control metric
    return weighted_metric / total_area

def _generate_FrameMapControlTileMetrics_object(map_control_values: FrameMapControlTileValues) -> FrameMapControlTileMetrics:
    """Generate FrameMapControlTileMetrics object from FrameMapControlTileValues object.

    Iterate through all tiles (both T and CT) with map control tiles values and
    create a map mapping from TileId to MapControlTileMetric

    For each tile, a map control metric is calculated with the below equation:
    sum(CT_values_for_tile) / (sum(CT_values_for_tile) + sum(T_values_for_tile))
    """
    tile_map_control_metrics = dict()

    all_tiles = set(map_control_values.t.keys() + map_control_values.ct.keys())
    for tile in all_tiles:
        ct_value = sum(map_control_values.ct.get(tile, [])
        t_value = sum(map_control_values.t.get(tile, [])

        all_tiles[tile] = ct_value / (t_value + ct_value)

    return all_tiles
