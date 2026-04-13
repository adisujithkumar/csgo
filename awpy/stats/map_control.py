from dataclasses import dataclass
from typing import TypeAlias
from collections import defaultdict

import polars as pl
import awpy
import awpy.data
import awpy.nav

# Module-level nav mesh cache to avoid repeated disk I/O.
_nav_cache: dict[str, awpy.nav.Nav] = {}


def _get_nav(map_name: str) -> awpy.nav.Nav:
    """Get Nav object for map, using module-level cache.

    Args:
        map_name: Name of the map (e.g. "de_dust2").

    Returns:
        Parsed Nav object.

    Raises:
        FileNotFoundError: If the navigation JSON file does not exist.
    """
    if map_name not in _nav_cache:
        nav_file_path = awpy.data.NAVS_DIR / f"{map_name}.json"
        if not nav_file_path.exists():
            msg = f"Navigation file not found: {nav_file_path}"
            raise FileNotFoundError(msg)
        _nav_cache[map_name] = awpy.nav.Nav.from_json(path=nav_file_path)
    return _nav_cache[map_name]


def clear_nav_cache() -> None:
    """Clear the nav mesh cache. Useful for testing or freeing memory."""
    _nav_cache.clear()


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
def get_tick_map_control_tile_values(map_name: str, tick: pl.dataframe.frame.DataFrame) -> FrameMapControlTileValues:
    """Get FrameMapControlTileValues object for a given frame."""
    # Validate inputs
    if not map_name:
        msg = "map_name cannot be empty"
        raise ValueError(msg)
    if tick.is_empty():
        msg = "tick DataFrame cannot be empty"
        raise ValueError(msg)

    # Get player positions for all alive players for each side
    # Note: Some demo data doesn't have 'armor' column, so we check if it exists
    if "armor" in tick.columns:
        t_alive_players = tick.filter(((pl.col("health") + pl.col("armor")) > 0) & (pl.col("side") == "t"))
        ct_alive_players = tick.filter(((pl.col("health") + pl.col("armor")) > 0) & (pl.col("side") == "ct"))
    else:
        # Fallback: use only health for alive check
        t_alive_players = tick.filter((pl.col("health") > 0) & (pl.col("side") == "t"))
        ct_alive_players = tick.filter((pl.col("health") > 0) & (pl.col("side") == "ct"))

    xyz_tuples_t = list(zip(t_alive_players['X'], t_alive_players['Y'], t_alive_players['Z']))
    xyz_tuples_ct = list(zip(ct_alive_players['X'], ct_alive_players['Y'], ct_alive_players['Z']))

    # Get map control tile values (TeamMapControlTileValues) for each side
    tick_mc_tile_values_t = _positions_to_map_control_values(map_name, xyz_tuples_t)
    tick_mc_tile_values_ct = _positions_to_map_control_values(map_name, xyz_tuples_ct)

    # Return FrameMapControlTileValues object with above data
    return FrameMapControlTileValues(t_values=tick_mc_tile_values_t, ct_values=tick_mc_tile_values_ct)

def get_map_control_tick_metric(map_name: str, tick: pl.dataframe.frame.DataFrame) -> MapControlFrameMetric:
    """Get MapControlFrameMetric for given frame."""
    # Get FrameMapControlTileValues objects for tiles
    tick_mc_tile_values = get_tick_map_control_tile_values(map_name, tick)

    # Get MapControlFrameMetric from FrameMapControlTileValues object
    tick_mc_metric = _calculate_map_control_frame_metric(tick_mc_tile_values, map_name)

    # Return MapControlFrameMetric
    return tick_mc_metric

def get_round_map_control_frame_metrics(map_name: str, _round: pl.dataframe.frame.DataFrame) -> list[MapControlFrameMetric]:
    """Get MapControlFrameMetric objects for each frame in a given round."""
    # Initialize an empty list
    results = []

    # Get FrameMapControlTileMetrics object from FrameMapControlTileValues object (see _generate_FrameMapControlTileMetrics_object for more context)

    # Iterate through each frame in the given round
    # For each frame, calculate the FrameMapControlMetric for that frame with get_map_control_frame_metric and add it to the list
    for tick_value, group_df in _round.group_by("tick", maintain_order=True):
        result = get_map_control_tick_metric(map_name, group_df)
        results.append(result)
    # Return list of floats (where the float at index i is the FrameMapControlMetric for frame i in the given round)
    return results

### Helper Functions
def _positions_to_map_control_values(map_name: str, player_positions: list[tuple[float]]) -> TeamMapControlTileValues:
    """Given list of player positions return TeamMapControlTileValues object."""
    # Handle empty player positions
    if not player_positions:
        return defaultdict(list)

    # Load nav data (cached)
    map_data = _get_nav(map_name)

    # Convert each passed in player position into NavArea area_id
    player_tiles = [find_closest_area(i, map_data) for i in player_positions]

    # Get map control values for each player's current tile
    team_MC_values = _bfs(player_tiles, map_name)

    # Return TeamMapControlTileValues object with above data
    return team_MC_values

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

    map_data = _get_nav(map_name)
    map_areas = map_data.areas

    for tile in source_tiles:
        tile_MC_values = _bfs_helper(tile, map_areas)
        returnTeamMapControlTileValues = combine_two_TeamMapControlTileValues(returnTeamMapControlTileValues, tile_MC_values)

    return returnTeamMapControlTileValues

def _bfs_helper(source_tile: TileId, area_info: dict) -> TeamMapControlTileValues:
    seen = set()
    returnPlayerMapControlTileValues = defaultdict(list)

    tileStack = [{"tileId": source_tile, "mcValue": 1, "lengthFromSource": 0}]

    while len(tileStack) > 0:
        currTile = tileStack.pop(0)
        currLength, nextlength = currTile["lengthFromSource"], currTile["lengthFromSource"] + 1
        currTileId = currTile['tileId']

        if currLength >= 10:
            continue

        returnPlayerMapControlTileValues[currTileId].append(currTile['mcValue'])
        seen.add(currTileId)

        currTileNeighbors = set(area_info[currTileId].connections)
        notSeenNeighbors = currTileNeighbors - seen
        for neighbor in notSeenNeighbors:
            tileStack.append({"tileId": neighbor, "mcValue": (10 - nextlength)/10, "lengthFromSource": nextlength})

    return returnPlayerMapControlTileValues

def _compute_polygon_area(corners):
    """Computes the area of a 3- or 4-sided polygon in the x-y plane using the shoelace formula.

    Args:
        corners (list of dict): Each dict has keys 'x', 'y', and 'z'.

    Returns:
        float: The absolute area of the polygon in 2D (x-y plane).
    """
    if len(corners) < 3 or len(corners) > 4:
        raise ValueError("This function only supports polygons with 3 or 4 corners.")

    # Extract (x, y) tuples
    points = [(corner.x, corner.y) for corner in corners]

    # Shoelace formula
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)

    return abs(area) / 2.0

def _calculate_map_control_frame_metric(map_control_values: FrameMapControlTileValues, map_name:str) -> MapControlFrameMetric:
    """Calculate MapControlFrameMetrics for given FrameMapControlTileValues object."""
    # Get a Map Control Tile Metric for each relevant tile
    frame_mc_tile_metrics = _generate_FrameMapControlTileMetrics_object(map_control_values)

    # Handle edge case: no tiles have values (e.g., all players dead)
    if not frame_mc_tile_metrics:
        has_t = bool(map_control_values.t_values)
        has_ct = bool(map_control_values.ct_values)
        if has_ct and not has_t:
            return 1.0  # Only CT alive → full CT control
        if has_t and not has_ct:
            return 0.0  # Only T alive → full T control
        return 0.5  # Both dead or both empty → neutral

    # Calculate a weighted average for the list of map control metrics where the weights is the area of the tiles
    map_data = _get_nav(map_name)
    map_areas = map_data.areas
    total_area = 0
    weighted_metric = 0
    for tile in frame_mc_tile_metrics.keys():
        tile_area = _compute_polygon_area(map_areas[tile].corners)
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

    all_tiles = set(list(map_control_values.t_values.keys()) + list(map_control_values.ct_values.keys()))
    for tile in all_tiles:
        ct_value = sum(map_control_values.ct_values.get(tile, []))
        t_value = sum(map_control_values.t_values.get(tile, []))

        total = t_value + ct_value
        if total == 0:
            continue

        tile_map_control_metrics[tile] = ct_value / total

    return tile_map_control_metrics

def point_in_polygon(x, y, polygon):
    """Ray casting algorithm for checking if a 2D point is inside a polygon."""
    n = len(polygon)
    inside = False
    px, py = x, y

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside

    return inside

def find_area_containing_point(xyz, area_data):
    """Finds the area_id containing the given (x, y, z) point, using both
    2D polygon containment and z-range filtering.

    Args:
        xyz (tuple): (x, y, z) coordinates of the point.
        area_data (dict): Dictionary containing 'areas' with 'corners'.

    Returns:
        int or None: area_id if found, otherwise None.
    """
    x, y, z = xyz

    for area in area_data.areas.values():
        corners = area.corners
        polygon_2d = [(corner.x, corner.y) for corner in corners]
        z_values = [corner.z for corner in corners]

        min_z, max_z = min(z_values), max(z_values)

        if min_z <= z <= max_z and point_in_polygon(x, y, polygon_2d):
            return area.area_id
    return None

import math

def euclidean_distance_3d(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0])**2 + 
                     (p1[1] - p2[1])**2 + 
                     (p1[2] - p2[2])**2)

def find_closest_area(xyz, area_data):
    """Finds the area whose corners are closest (min Euclidean distance) to the point.

    Args:
        xyz (tuple): (x, y, z) coordinates of the point.
        area_data (dict): Dictionary with 'areas', each having 'corners'.

    Returns:
        int or None: area_id of the closest area, or None if no areas exist.
    """
    closest_area_id = None
    min_distance = float('inf')

    for area in area_data.areas.values():
        for corner in area.corners:
            corner_point = (corner.x, corner.y, corner.z)
            dist = euclidean_distance_3d(xyz, corner_point)

            if dist < min_distance:
                min_distance = dist
                closest_area_id = area.area_id

    return closest_area_id
