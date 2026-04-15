"""Map control analytics using BFS over the navigation mesh.

This module computes per-tile, per-frame, and per-round map control metrics
for CS2 demos. The core idea: from each alive player's nav tile, propagate a
"control value" to nearby tiles, combine across the team, and convert to a
ratio CT / (CT + T) per tile. A frame metric is the polygon-area-weighted
average across tiles.

Two propagation models are available:

1. **Circular BFS** (default / legacy)
   Equal expansion in all directions, value = (depth - hops) / depth.
   Simple and fast, but does not model facing direction or weapon type.
   Ignores the fact that a player holding an AWP at mid does not threaten
   tiles behind them or tiles on the opposite side of the map.

2. **Conical BFS** (threat-aware)
   Directional cone centered on the player's yaw, with depth and half-angle
   that depend on weapon class. The value on each tile is the product of a
   range term (hops-based linear falloff) and an angular term (cosine^k
   inside the cone, zero outside). An optional rear halo captures the fact
   that players still have some peripheral / audio awareness of space
   directly behind them, but at a heavily attenuated weight.

Conical BFS is enabled by passing ``player_yaws`` and ``player_weapons``
through the public entry points. When these are missing, the code falls
back to circular BFS so existing tests and callers keep working unchanged.

Weapon-class cone parameters (``WEAPON_CONE_PARAMS``) were derived from
1. CS2 weapon accuracy tables (``inaccuracy_first_shot`` / accurate range)
2. Empirical kill-distance data from HLTV demos in
   ``output/full_dust2/cs2_analytics.duckdb`` (``kills`` table).
See ``WEAPON_CONE_PARAMS`` docstring for the full derivation.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, TypeAlias

import polars as pl

import awpy
import awpy.data
import awpy.nav

# ---------------------------------------------------------------------------
# Nav mesh cache
# ---------------------------------------------------------------------------

# Module-level nav mesh cache to avoid repeated disk I/O.
_nav_cache: dict[str, awpy.nav.Nav] = {}

# Per-map centroid cache: maps tile_id -> (cx, cy).
_centroid_cache: dict[str, dict[int, tuple[float, float]]] = {}

# Per-map Z-centroid cache: maps tile_id -> mean_z (for Z-floor gating).
_z_centroid_cache: dict[str, dict[int, float]] = {}


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
    """Clear the nav mesh and centroid caches. Useful for tests / freeing memory."""
    _nav_cache.clear()
    _centroid_cache.clear()
    _z_centroid_cache.clear()
    _conical_bfs_cached.cache_clear()


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Weapon classification and cone parameters
# ---------------------------------------------------------------------------

# Weapon classes used by the conical BFS. Each class has its own cone geometry.
WEAPON_CLASSES = ("awp", "rifle", "smg", "pistol", "shotgun", "knife")

# Weapon name → class.
#
# Names match the ``weapon`` column in demoparser2's kills table (lowercase,
# e.g. ``ak47``, ``m4a1_silencer``, ``usp_silencer``). Unknown weapons default
# to ``rifle`` (a conservative middle ground).
_WEAPON_NAME_TO_CLASS: dict[str, str] = {
    # Sniper rifles (awp class) — both internal and demoparser2 display names
    "awp": "awp",
    "ssg08": "awp",
    "scar20": "awp",
    "g3sg1": "awp",
    "ssg 08": "awp",       # display name
    "scar-20": "awp",      # display name
    # Automatic rifles — internal names
    "ak47": "rifle",
    "m4a1": "rifle",
    "m4a1_silencer": "rifle",
    "m4a4": "rifle",
    "aug": "rifle",
    "sg556": "rifle",
    "sg553": "rifle",
    "famas": "rifle",
    "galilar": "rifle",
    "m249": "rifle",
    "negev": "rifle",
    # Automatic rifles — demoparser2 display names
    "ak-47": "rifle",
    "m4a1-s": "rifle",
    "galil ar": "rifle",
    "sg 553": "rifle",
    # SMGs — internal names
    "mp9": "smg",
    "mac10": "smg",
    "mp7": "smg",
    "mp5sd": "smg",
    "ump45": "smg",
    "p90": "smg",
    "bizon": "smg",
    # SMGs — demoparser2 display names
    "mac-10": "smg",
    "mp5-sd": "smg",
    "ump-45": "smg",
    "pp-bizon": "smg",
    # Pistols — internal names
    "glock": "pistol",
    "hkp2000": "pistol",
    "usp_silencer": "pistol",
    "usp_silencer_off": "pistol",
    "p250": "pistol",
    "fiveseven": "pistol",
    "tec9": "pistol",
    "cz75a": "pistol",
    "cz75": "pistol",
    "deagle": "pistol",
    "revolver": "pistol",
    "elite": "pistol",
    # Pistols — demoparser2 display names
    "glock-18": "pistol",
    "usp-s": "pistol",
    "p2000": "pistol",
    "five-seven": "pistol",
    "tec-9": "pistol",
    "desert eagle": "pistol",
    "cz75-auto": "pistol",
    "dual berettas": "pistol",
    "r8 revolver": "pistol",
    # Shotguns — internal names
    "nova": "shotgun",
    "xm1014": "shotgun",
    "mag7": "shotgun",
    "sawedoff": "shotgun",
    # Shotguns — demoparser2 display names
    "mag-7": "shotgun",
    "sawed-off": "shotgun",
    # Knives — internal names
    "knife": "knife",
    "knife_t": "knife",
    "knife_ct": "knife",
    "taser": "knife",
    "zeus": "knife",
    # Knives — demoparser2 display names (cosmetic skins all map to knife)
    "butterfly knife": "knife",
    "karambit": "knife",
    "m9 bayonet": "knife",
    "bayonet": "knife",
    "flip knife": "knife",
    "gut knife": "knife",
    "falchion knife": "knife",
    "shadow daggers": "knife",
    "huntsman knife": "knife",
    "bowie knife": "knife",
    "navaja knife": "knife",
    "stiletto knife": "knife",
    "talon knife": "knife",
    "ursus knife": "knife",
    "nomad knife": "knife",
    "skeleton knife": "knife",
    "paracord knife": "knife",
    "survival knife": "knife",
    "classic knife": "knife",
    "kukri knife": "knife",
    "zeus x27": "knife",
    # Utility items (active weapon when throwing/planting — no sustained threat)
    # Treat as "knife" so they get the 360° short-range halo, not rifle-class cone.
    "smoke grenade": "knife",
    "flashbang": "knife",
    "molotov": "knife",
    "incendiary grenade": "knife",
    "decoy grenade": "knife",
    "c4 explosive": "knife",
    "high explosive grenade": "knife",
    "he grenade": "knife",
}


def _classify_weapon(weapon: str | int | float | None) -> str:
    """Classify a weapon into one of ``WEAPON_CLASSES``.

    Accepts either:

    - A string weapon name (e.g. ``"ak47"``, ``"awp"``, ``"knife_butterfly"``).
      Match is case-insensitive and any ``knife_*`` variant maps to ``knife``.
    - A numeric ``current_equip_value`` used as a fuzzy tier. This is noisy
      because equip value is the *total* inventory value (primary + secondary
      + armor + utility), but is useful as a fallback when only equip value
      is available per tick.

    Args:
        weapon: Weapon name, equip value, or None.

    Returns:
        One of ``WEAPON_CLASSES``. Defaults to ``"rifle"`` for unknowns.
    """
    if weapon is None:
        return "rifle"
    if isinstance(weapon, str):
        name = weapon.lower().strip()
        if not name:
            return "rifle"
        # Catch any knife variant: internal (knife_butterfly, knife_t …) and
        # display names ("Butterfly Knife", "Kukri Knife", "Shadow Daggers" …)
        if name.startswith("knife") or name.endswith("knife") or "bayonet" in name or "daggers" in name:
            return "knife"
        if name in _WEAPON_NAME_TO_CLASS:
            return _WEAPON_NAME_TO_CLASS[name]
        # Not a known weapon name — try to parse as a numeric equip value
        # (this is how ``_extract_player_state`` forwards ``current_equip_value``
        # when no explicit ``weapon`` column is present).
        try:
            eq = float(name)
        except ValueError:
            return "rifle"
    else:
        try:
            eq = float(weapon)
        except (TypeError, ValueError):
            return "rifle"
    # Thresholds chosen from kill data: awp median=7050, rifle median=5000,
    # smg median=3350, pistol median≈1000-2000, knife≈650.
    if eq >= 6200:
        return "awp"
    if eq >= 4000:
        return "rifle"
    if eq >= 2200:
        return "smg"
    if eq >= 700:
        return "pistol"
    return "knife"


# Cone parameters per weapon class.
#
# Keys:
#   max_depth        — BFS hop count. On dust2 a hop is ~1 meter median, so
#                      depth ≈ effective range in meters.
#   half_angle_deg   — half the cone's horizontal opening angle in degrees.
#   falloff_exponent — cosine^k exponent. Larger k = sharper focus at yaw.
#   rear_depth       — hops for rear-halo (awareness behind player). 0 = off.
#   rear_multiplier  — weight of rear halo tiles relative to forward max.
#
# Derivation of max_depth (rounded up from median kill distance on dust2):
#   AWP      median 27.2m  → 20 hops (AWP threatens further than it kills)
#   Rifle    median 18-19m → 15 hops
#   SMG      median 12-14m → 10 hops
#   Pistol   median 16-20m → 12 hops (deagle skews this up)
#   Shotgun  median 5-10m  →  5 hops
#   Knife    median  1.2m  →  2 hops
#
# Derivation of half_angle_deg: scoped snipers hold single tight lines → 15°.
# Rifles ADS cone and rotate to engage → 40°. SMGs hipfire close-quarters → 55°.
# Pistols are rifle-adjacent → 40°. Shotguns effective only at very close
# quarters but in ANY direction → 75°. Knives are effectively 360° melee → 180°.
#
# Derivation of falloff_exponent: AWP wants a very sharp "laser" feel (k=4),
# rifles a moderate focus (k=2), SMG/pistol/shotgun a gentle taper (k=1.5),
# knife uniform (k=0 → all-ones within cone). Higher k = more narrow effective.
WEAPON_CONE_PARAMS: dict[str, dict] = {
    "awp": {
        "max_depth": 20,
        "half_angle_deg": 15.0,
        "falloff_exponent": 4.0,
        "rear_depth": 2,
        "rear_multiplier": 0.10,
    },
    "rifle": {
        "max_depth": 15,
        "half_angle_deg": 40.0,
        "falloff_exponent": 2.0,
        "rear_depth": 3,
        "rear_multiplier": 0.15,
    },
    "smg": {
        "max_depth": 10,
        "half_angle_deg": 55.0,
        "falloff_exponent": 1.5,
        "rear_depth": 3,
        "rear_multiplier": 0.20,
    },
    "pistol": {
        "max_depth": 12,
        "half_angle_deg": 40.0,
        "falloff_exponent": 2.0,
        "rear_depth": 3,
        "rear_multiplier": 0.15,
    },
    "shotgun": {
        "max_depth": 5,
        "half_angle_deg": 75.0,
        "falloff_exponent": 1.5,
        "rear_depth": 2,
        "rear_multiplier": 0.25,
    },
    "knife": {
        "max_depth": 2,
        "half_angle_deg": 180.0,
        "falloff_exponent": 0.0,
        "rear_depth": 0,
        "rear_multiplier": 0.0,
    },
}


# ---------------------------------------------------------------------------
# MapControlConfig (feature flags + tunables)
# ---------------------------------------------------------------------------


@dataclass
class MapControlConfig:
    """Configuration for map control computation.

    All feature flags default to current (circular BFS) behavior. To enable
    the directional threat model, pass ``use_conical_bfs=True`` AND provide
    ``player_yaws`` / ``player_weapons`` when calling the public functions.

    Attributes:
        bfs_depth: Max BFS hops for circular BFS.
        use_conical_bfs: Enable the directional weapon-aware cone model.
            When True, ``player_yaws`` and ``player_weapons`` must be passed
            through the public API or we silently fall back to circular BFS.
        yaw_bucket_deg: Yaw is discretized to this many degrees for caching
            so that tiny per-tick wiggles do not bust the cone-BFS LRU cache.
            10° is a good default.
        cone_params_override: Optional per-class overrides of
            ``WEAPON_CONE_PARAMS`` (e.g. to tune depth for one weapon at runtime).
        rear_halo_enabled: Enable the small rear-awareness halo (see
            ``WEAPON_CONE_PARAMS`` rear_depth / rear_multiplier).
        smoke_blocking_enabled: (future) Block BFS propagation through smokes.
        molotov_denial_enabled: (future) Down-weight molotov-covered tiles.
        flash_weight_enabled: (future) Scale player contribution by flash state.
        sightline_enabled: (future) Use visibility checker for line-of-sight gate.
        weapon_weight_enabled: (future) Multiply contribution by weapon "impact".
    """

    # BFS parameters
    bfs_depth: int = 10

    # Conical BFS
    use_conical_bfs: bool = False
    yaw_bucket_deg: float = 10.0
    cone_params_override: dict[str, dict] | None = None
    rear_halo_enabled: bool = True
    # Z-floor gate: skip cone coverage to tiles whose mean-Z differs from the
    # source tile by more than this many source units. Fixes nuke upper/lower
    # bleed where 2D bearing matches across stacked floors (dz≈224 units).
    # Set to 0 (default) to disable. Recommended value for nuke: 150.
    z_floor_threshold: float = 0.0

    # Utility integration (future)
    smoke_blocking_enabled: bool = False
    smoke_radius: float = 144.0
    molotov_denial_enabled: bool = False
    molotov_radius: float = 96.0
    flash_weight_enabled: bool = False
    flash_weight_multiplier: float = 0.1

    # Sightline (future)
    sightline_enabled: bool = False
    sightline_forward_angle: float = 90.0
    sightline_rear_multiplier: float = 0.5

    # Weapon weighting (future)
    weapon_weight_enabled: bool = False

    def cone_params_for(self, weapon_class: str) -> dict:
        """Return cone parameters for a weapon class, honoring overrides."""
        base = WEAPON_CONE_PARAMS.get(weapon_class, WEAPON_CONE_PARAMS["rifle"])
        if self.cone_params_override and weapon_class in self.cone_params_override:
            return {**base, **self.cone_params_override[weapon_class]}
        return base


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def get_tick_map_control_tile_values(
    map_name: str,
    tick: pl.dataframe.frame.DataFrame,
    config: MapControlConfig | None = None,
) -> FrameMapControlTileValues:
    """Get FrameMapControlTileValues object for a given frame.

    Args:
        map_name: Map name (e.g. ``"de_dust2"``).
        tick: Polars DataFrame of a single tick's player rows. Must contain
            at minimum: ``X``, ``Y``, ``Z``, ``side``, ``health``. If
            ``config.use_conical_bfs`` is True, ``yaw`` is required and
            either ``weapon`` (string) or ``current_equip_value`` (float) is
            used to classify weapon class.
        config: Optional ``MapControlConfig``. If None, legacy circular BFS
            with depth 10 is used, exactly matching pre-cone behavior.

    Returns:
        ``FrameMapControlTileValues`` with both teams' tile values.
    """
    if config is None:
        config = MapControlConfig()

    # Validate inputs
    if not map_name:
        msg = "map_name cannot be empty"
        raise ValueError(msg)
    if tick.is_empty():
        msg = "tick DataFrame cannot be empty"
        raise ValueError(msg)

    # Alive filter. Some demo snapshots lack "armor" — fall back to health only.
    if "armor" in tick.columns:
        alive_mask = (pl.col("health") + pl.col("armor")) > 0
    else:
        alive_mask = pl.col("health") > 0

    t_alive_players = tick.filter(alive_mask & (pl.col("side") == "t"))
    ct_alive_players = tick.filter(alive_mask & (pl.col("side") == "ct"))

    t_positions, t_yaws, t_weapons = _extract_player_state(t_alive_players, config)
    ct_positions, ct_yaws, ct_weapons = _extract_player_state(ct_alive_players, config)

    tick_mc_tile_values_t = _positions_to_map_control_values(
        map_name, t_positions, player_yaws=t_yaws, player_weapons=t_weapons, config=config
    )
    tick_mc_tile_values_ct = _positions_to_map_control_values(
        map_name, ct_positions, player_yaws=ct_yaws, player_weapons=ct_weapons, config=config
    )

    return FrameMapControlTileValues(t_values=tick_mc_tile_values_t, ct_values=tick_mc_tile_values_ct)


def get_map_control_tick_metric(
    map_name: str,
    tick: pl.dataframe.frame.DataFrame,
    config: MapControlConfig | None = None,
) -> MapControlFrameMetric:
    """Get MapControlFrameMetric for given frame."""
    if config is None:
        config = MapControlConfig()
    tick_mc_tile_values = get_tick_map_control_tile_values(map_name, tick, config=config)
    return _calculate_map_control_frame_metric(tick_mc_tile_values, map_name)


def get_round_map_control_frame_metrics(
    map_name: str,
    _round: pl.dataframe.frame.DataFrame,
    config: MapControlConfig | None = None,
) -> list[MapControlFrameMetric]:
    """Get MapControlFrameMetric objects for each frame in a given round."""
    if config is None:
        config = MapControlConfig()
    results: list[MapControlFrameMetric] = []
    for _tick_value, group_df in _round.group_by("tick", maintain_order=True):
        results.append(get_map_control_tick_metric(map_name, group_df, config=config))
    return results


# ---------------------------------------------------------------------------
# State extraction and position → control values plumbing
# ---------------------------------------------------------------------------


def _extract_player_state(
    players: pl.DataFrame,
    config: MapControlConfig,
) -> tuple[list[tuple[float, float, float]], list[float] | None, list[str] | None]:
    """Pull (positions, yaws, weapons) out of a per-team DataFrame.

    Returns yaws/weapons as ``None`` when conical BFS is disabled OR when the
    required columns are missing — the caller will then fall back to circular.
    """
    positions = list(zip(players["X"], players["Y"], players["Z"]))

    if not config.use_conical_bfs:
        return positions, None, None

    yaws: list[float] | None = None
    if "yaw" in players.columns:
        yaws = [float(y) if y is not None else 0.0 for y in players["yaw"]]

    weapons: list[str] | None = None
    if "weapon" in players.columns:
        weapons = [w if w is not None else "" for w in players["weapon"]]
    elif "current_equip_value" in players.columns:
        # Numeric fallback — _classify_weapon handles floats.
        weapons = [str(v) if v is not None else "" for v in players["current_equip_value"]]

    return positions, yaws, weapons


def _positions_to_map_control_values(
    map_name: str,
    player_positions: Sequence[tuple[float, float, float]],
    *,
    player_yaws: Sequence[float] | None = None,
    player_weapons: Sequence[str | float] | None = None,
    config: MapControlConfig | None = None,
) -> TeamMapControlTileValues:
    """Given player positions (and optional yaws/weapons), compute tile values.

    When ``config.use_conical_bfs`` is True AND ``player_yaws`` is provided,
    the directional cone model is used. Otherwise, the legacy circular BFS
    is used so all existing tests and callers are unaffected.
    """
    if config is None:
        config = MapControlConfig()

    if not player_positions:
        return defaultdict(list)

    map_data = _get_nav(map_name)
    player_tiles = [find_closest_area(pos, map_data) for pos in player_positions]

    if config.use_conical_bfs and player_yaws is not None:
        if player_weapons is None:
            player_weapons = ["rifle"] * len(player_tiles)
        return _conical_bfs_team(
            player_tiles=player_tiles,
            player_yaws=player_yaws,
            player_weapons=player_weapons,
            map_name=map_name,
            config=config,
        )

    return _bfs(player_tiles, map_name, max_depth=config.bfs_depth)


def combine_two_TeamMapControlTileValues(dict1, dict2):
    combined: dict[int, list[float]] = {}
    for key, val in dict1.items():
        combined[key] = list(val)
    for key, val in dict2.items():
        if key in combined:
            combined[key].extend(val)
        else:
            combined[key] = list(val)
    return combined


# ---------------------------------------------------------------------------
# Circular (legacy) BFS
# ---------------------------------------------------------------------------


def _bfs(source_tiles: list[TileId], map_name: str, max_depth: int = 10) -> TeamMapControlTileValues:
    """Given list of TileIds return TeamMapControlTileValues object (circular)."""
    result: dict[int, list[float]] = defaultdict(list)
    map_areas = _get_nav(map_name).areas
    for tile in source_tiles:
        tile_values = _bfs_helper(tile, map_areas, max_depth=max_depth)
        result = combine_two_TeamMapControlTileValues(result, tile_values)
    return result


def _bfs_helper(
    source_tile: TileId,
    area_info: dict,
    max_depth: int = 10,
) -> TeamMapControlTileValues:
    """Circular BFS from a single source tile. Matches legacy semantics."""
    seen: set[int] = set()
    values: dict[int, list[float]] = defaultdict(list)

    queue: deque[dict] = deque(
        [{"tileId": source_tile, "mcValue": 1, "lengthFromSource": 0}]
    )

    while queue:
        curr = queue.popleft()
        curr_length = curr["lengthFromSource"]
        next_length = curr_length + 1
        curr_tile_id = curr["tileId"]

        if curr_length >= max_depth:
            continue

        values[curr_tile_id].append(curr["mcValue"])
        seen.add(curr_tile_id)

        neighbors = set(area_info[curr_tile_id].connections)
        for neighbor in neighbors - seen:
            queue.append(
                {
                    "tileId": neighbor,
                    "mcValue": (max_depth - next_length) / max_depth,
                    "lengthFromSource": next_length,
                }
            )

    return values


# ---------------------------------------------------------------------------
# Tile centroid helpers
# ---------------------------------------------------------------------------


def _compute_tile_centroids(nav: awpy.nav.Nav) -> dict[int, tuple[float, float]]:
    """Compute (x, y) centroid for every nav tile as the mean of its corners."""
    out: dict[int, tuple[float, float]] = {}
    for area_id, area in nav.areas.items():
        n = len(area.corners)
        if n == 0:
            out[area_id] = (0.0, 0.0)
            continue
        cx = sum(c.x for c in area.corners) / n
        cy = sum(c.y for c in area.corners) / n
        out[area_id] = (cx, cy)
    return out


def _compute_tile_z_centroids(nav: awpy.nav.Nav) -> dict[int, float]:
    """Compute mean Z centroid for every nav tile.

    Used by the Z-floor gate in ``_conical_bfs`` to prevent the 2D bearing
    check from bleeding cone coverage across stacked floors (e.g. nuke upper
    site → sub-ledge at dz≈224 units).
    """
    out: dict[int, float] = {}
    for area_id, area in nav.areas.items():
        corners = area.corners
        if not corners:
            out[area_id] = 0.0
        else:
            out[area_id] = sum(c.z for c in corners) / len(corners)
    return out


def _get_tile_centroids(map_name: str) -> dict[int, tuple[float, float]]:
    """Cached accessor for per-map (x, y) tile centroids."""
    if map_name not in _centroid_cache:
        _centroid_cache[map_name] = _compute_tile_centroids(_get_nav(map_name))
    return _centroid_cache[map_name]


def _get_tile_z_centroids(map_name: str) -> dict[int, float]:
    """Cached accessor for per-map mean-Z tile centroids."""
    if map_name not in _z_centroid_cache:
        _z_centroid_cache[map_name] = _compute_tile_z_centroids(_get_nav(map_name))
    return _z_centroid_cache[map_name]


# ---------------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------------


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference in degrees, handling 0/360 wrap.

    Args:
        a: First angle in degrees.
        b: Second angle in degrees.

    Returns:
        Absolute difference in [0, 180].
    """
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def _tile_direction_deg(
    source: tuple[float, float],
    target: tuple[float, float],
) -> float | None:
    """Compass-style bearing in degrees from ``source`` to ``target``.

    CS2's ``yaw`` is measured with 0° = +X axis, increasing counter-clockwise
    (standard math convention, matching ``atan2(dy, dx)`` in degrees).
    Returns None when the points coincide (undefined direction).
    """
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    if dx * dx + dy * dy < 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def _angle_weight(
    angle_from_facing_deg: float,
    half_angle_deg: float,
    falloff_exponent: float,
) -> float:
    """Cosine^k angular falloff on [0, half_angle_deg].

    Rescales ``angle_from_facing`` so that 0° maps to cos(0)=1 and exactly
    ``half_angle`` maps to cos(π/2)=0. Raising to ``falloff_exponent`` sharpens
    the peak for scoped weapons (higher k = narrower effective cone).
    """
    if angle_from_facing_deg >= half_angle_deg:
        return 0.0
    if half_angle_deg <= 0.0:
        return 0.0
    # Map angle ∈ [0, half_angle] → [0, π/2]
    scaled = (angle_from_facing_deg / half_angle_deg) * (math.pi / 2.0)
    base = math.cos(scaled)
    if falloff_exponent == 0.0:
        return 1.0 if base > 0.0 else 0.0
    return max(0.0, base) ** falloff_exponent


# ---------------------------------------------------------------------------
# Conical BFS
# ---------------------------------------------------------------------------


def _conical_bfs(
    source_tile: TileId,
    player_yaw_deg: float,
    weapon_class: str,
    area_info: dict,
    tile_centroids: dict[int, tuple[float, float]],
    params: dict,
    rear_halo_enabled: bool = True,
    tile_z: dict[int, float] | None = None,
    z_floor_threshold: float = 0.0,
) -> dict[int, float]:
    """Directional BFS from a single source tile.

    Value formula:

        range_weight = max(0, (max_depth - hops) / max_depth)
        angle_from_facing = |angle_diff(tile_direction, player_yaw)|
        angle_weight = (cos(angle_from_facing scaled to [0, π/2]))^k
        value = range_weight * angle_weight   (forward cone)
              = rear_multiplier * range_weight  (rear halo, if enabled)

    Source tile always gets value = 1.0 regardless of direction.

    Args:
        source_tile: Tile ID the player is standing on.
        player_yaw_deg: Player facing angle in degrees (CS2 convention).
        weapon_class: One of ``WEAPON_CLASSES``. Used only for logging —
            caller passes ``params`` explicitly.
        area_info: ``nav.areas`` dict (tile_id → NavArea).
        tile_centroids: map of tile_id → (x, y) centroid, pre-computed.
        params: Cone parameters (see ``WEAPON_CONE_PARAMS``).
        rear_halo_enabled: If True, apply a small rear awareness halo.
        tile_z: Optional map of tile_id → mean Z. When provided and
            ``z_floor_threshold > 0``, neighbors whose mean Z differs from
            the source by more than the threshold are skipped. Prevents cone
            bleeding across stacked floors on maps like nuke.
        z_floor_threshold: Source-unit Z difference above which a neighbor
            tile is considered on a different floor. 0 = disabled.

    Returns:
        Dict mapping tile_id → float value in [0, 1].
    """
    max_depth: int = int(params["max_depth"])
    half_angle: float = float(params["half_angle_deg"])
    falloff_exp: float = float(params["falloff_exponent"])
    rear_depth: int = int(params.get("rear_depth", 0))
    rear_mult: float = float(params.get("rear_multiplier", 0.0))

    if max_depth <= 0 or source_tile not in area_info:
        return {source_tile: 1.0} if source_tile in area_info else {}

    source_xy = tile_centroids[source_tile]
    source_z = tile_z[source_tile] if (tile_z and z_floor_threshold > 0.0) else 0.0
    values: dict[int, float] = {source_tile: 1.0}
    forward_best: dict[int, float] = {}
    rear_best: dict[int, float] = {}

    # BFS — but we compute angle from the SOURCE to each tile's centroid rather
    # than path-cumulative angles, because threat is about sightline direction
    # from the player, not the nav-mesh walk direction.
    visited: set[int] = {source_tile}
    queue: deque[tuple[int, int]] = deque([(source_tile, 0)])

    while queue:
        tile_id, hops = queue.popleft()
        next_hops = hops + 1

        area = area_info.get(tile_id)
        if area is None:
            continue

        for neighbor in area.connections:
            if neighbor in visited or neighbor not in tile_centroids:
                continue

            # Z-floor gate: skip tiles on a different vertical level.
            # This prevents the 2D bearing check from bleeding cone coverage
            # across stacked nav areas (e.g. nuke upper → sub-ledge, dz≈224).
            if tile_z and z_floor_threshold > 0.0:
                neighbor_z = tile_z.get(neighbor, source_z)
                if abs(neighbor_z - source_z) > z_floor_threshold:
                    continue

            visited.add(neighbor)

            # Range term — linear falloff in hops, uses the larger of the
            # forward and rear depths so the same BFS walk populates both.
            max_reach = max(max_depth, rear_depth if rear_halo_enabled else 0)
            if next_hops > max_reach:
                continue

            direction = _tile_direction_deg(source_xy, tile_centroids[neighbor])
            if direction is None:
                angle = 0.0
            else:
                angle = _angle_diff(direction, player_yaw_deg)

            # Forward cone contribution
            if next_hops <= max_depth and angle < half_angle:
                range_w = max(0.0, (max_depth - next_hops) / max_depth)
                angle_w = _angle_weight(angle, half_angle, falloff_exp)
                forward_val = range_w * angle_w
                if forward_val > 0.0:
                    prev = forward_best.get(neighbor, 0.0)
                    if forward_val > prev:
                        forward_best[neighbor] = forward_val

            # Rear halo contribution (small, uniform, shorter depth)
            if (
                rear_halo_enabled
                and rear_depth > 0
                and next_hops <= rear_depth
                and angle >= half_angle
            ):
                range_w = max(0.0, (rear_depth - next_hops) / rear_depth)
                rear_val = rear_mult * range_w
                if rear_val > 0.0:
                    prev = rear_best.get(neighbor, 0.0)
                    if rear_val > prev:
                        rear_best[neighbor] = rear_val

            queue.append((neighbor, next_hops))

    # Merge: forward cone takes precedence; otherwise use rear halo.
    for tile_id, val in forward_best.items():
        if val > values.get(tile_id, 0.0):
            values[tile_id] = val
    for tile_id, val in rear_best.items():
        if val > values.get(tile_id, 0.0):
            values[tile_id] = val

    return values


# LRU-cached variant keyed on (map_name, source_tile, yaw_bucket, weapon_class,
# rear_halo_enabled, override_hash). Stores up to 100k entries.
@lru_cache(maxsize=100_000)
def _conical_bfs_cached(
    map_name: str,
    source_tile: int,
    yaw_bucket: int,
    weapon_class: str,
    rear_halo_enabled: bool,
    params_key: tuple,
    z_floor_threshold_int: int = 0,
) -> tuple[tuple[int, float], ...]:
    """Cacheable wrapper. ``params_key`` is a hashable tuple of cone params.

    ``z_floor_threshold_int`` is the threshold rounded to the nearest integer
    so it remains hashable. 0 means the Z-floor gate is disabled.
    """
    nav = _get_nav(map_name)
    centroids = _get_tile_centroids(map_name)
    tile_z = _get_tile_z_centroids(map_name) if z_floor_threshold_int > 0 else None
    # Reconstruct params dict from the hashable key.
    params = {
        "max_depth": params_key[0],
        "half_angle_deg": params_key[1],
        "falloff_exponent": params_key[2],
        "rear_depth": params_key[3],
        "rear_multiplier": params_key[4],
    }
    # Use bucket CENTER as the yaw we feed to the BFS, not the raw yaw.
    yaw_center = yaw_bucket * 10.0  # matches default bucket width of 10°
    result = _conical_bfs(
        source_tile=source_tile,
        player_yaw_deg=yaw_center,
        weapon_class=weapon_class,
        area_info=nav.areas,
        tile_centroids=centroids,
        params=params,
        rear_halo_enabled=rear_halo_enabled,
        tile_z=tile_z,
        z_floor_threshold=float(z_floor_threshold_int),
    )
    return tuple(sorted(result.items()))


def _conical_bfs_team(
    player_tiles: Sequence[int | None],
    player_yaws: Sequence[float],
    player_weapons: Sequence[str | float],
    map_name: str,
    config: MapControlConfig,
) -> TeamMapControlTileValues:
    """Run the cone BFS for every alive player on one team and merge results."""
    result: dict[int, list[float]] = defaultdict(list)

    bucket_width = config.yaw_bucket_deg if config.yaw_bucket_deg > 0 else 10.0

    for tile, yaw, weapon in zip(player_tiles, player_yaws, player_weapons):
        if tile is None:
            continue
        weapon_class = _classify_weapon(weapon)
        params = config.cone_params_for(weapon_class)

        # Bucket yaw for caching. The cache stores the bucket-CENTER result,
        # and we use 10° buckets internally (rewriting this needs a helper
        # that agrees with _conical_bfs_cached).
        yaw_norm = yaw % 360.0
        yaw_bucket = int(round(yaw_norm / bucket_width))

        # Hashable params tuple used as cache key.
        params_key = (
            int(params["max_depth"]),
            float(params["half_angle_deg"]),
            float(params["falloff_exponent"]),
            int(params.get("rear_depth", 0)),
            float(params.get("rear_multiplier", 0.0)),
        )

        z_thresh_int = int(round(config.z_floor_threshold))

        # Per-player cone BFS. We always go through the cached wrapper using
        # 10° default bucket width. If the user configures a different
        # ``yaw_bucket_deg`` we bypass the cache and compute directly so the
        # semantic stays consistent.
        if bucket_width == 10.0:
            tiles_tuple = _conical_bfs_cached(
                map_name=map_name,
                source_tile=tile,
                yaw_bucket=yaw_bucket,
                weapon_class=weapon_class,
                rear_halo_enabled=config.rear_halo_enabled,
                params_key=params_key,
                z_floor_threshold_int=z_thresh_int,
            )
            tile_values = dict(tiles_tuple)
        else:
            nav = _get_nav(map_name)
            centroids = _get_tile_centroids(map_name)
            tile_z = _get_tile_z_centroids(map_name) if z_thresh_int > 0 else None
            tile_values = _conical_bfs(
                source_tile=tile,
                player_yaw_deg=yaw_norm,
                weapon_class=weapon_class,
                area_info=nav.areas,
                tile_centroids=centroids,
                params=params,
                rear_halo_enabled=config.rear_halo_enabled,
                tile_z=tile_z,
                z_floor_threshold=config.z_floor_threshold,
            )

        for t_id, val in tile_values.items():
            if val > 0.0:
                result[t_id].append(val)

    return result


# ---------------------------------------------------------------------------
# Geometry helpers (polygon area, point-in-polygon, nearest area)
# ---------------------------------------------------------------------------


def _compute_polygon_area(corners):
    """Computes the area of a 3- or 4-sided polygon in the x-y plane (shoelace)."""
    if len(corners) < 3 or len(corners) > 4:
        raise ValueError("This function only supports polygons with 3 or 4 corners.")
    points = [(corner.x, corner.y) for corner in corners]
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _calculate_map_control_frame_metric(
    map_control_values: FrameMapControlTileValues, map_name: str
) -> MapControlFrameMetric:
    """Calculate MapControlFrameMetric (polygon-area-weighted) for a frame."""
    frame_mc_tile_metrics = _generate_FrameMapControlTileMetrics_object(map_control_values)

    if not frame_mc_tile_metrics:
        has_t = bool(map_control_values.t_values)
        has_ct = bool(map_control_values.ct_values)
        if has_ct and not has_t:
            return 1.0
        if has_t and not has_ct:
            return 0.0
        return 0.5

    map_areas = _get_nav(map_name).areas
    total_area = 0.0
    weighted_metric = 0.0
    for tile, metric in frame_mc_tile_metrics.items():
        tile_area = _compute_polygon_area(map_areas[tile].corners)
        weighted_metric += tile_area * metric
        total_area += tile_area
    return weighted_metric / total_area if total_area > 0 else 0.5


def _generate_FrameMapControlTileMetrics_object(
    map_control_values: FrameMapControlTileValues,
) -> FrameMapControlTileMetrics:
    """Per-tile CT / (CT + T) ratio across the frame."""
    tile_map_control_metrics: dict[int, float] = {}
    all_tiles = set(
        list(map_control_values.t_values.keys()) + list(map_control_values.ct_values.keys())
    )
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
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi
        ):
            inside = not inside
    return inside


def find_area_containing_point(xyz, area_data):
    """Finds the area_id containing a 3D point via polygon containment + z range."""
    x, y, z = xyz
    for area in area_data.areas.values():
        corners = area.corners
        polygon_2d = [(corner.x, corner.y) for corner in corners]
        z_values = [corner.z for corner in corners]
        min_z, max_z = min(z_values), max(z_values)
        if min_z <= z <= max_z and point_in_polygon(x, y, polygon_2d):
            return area.area_id
    return None


def euclidean_distance_3d(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )


def find_closest_area(xyz, area_data):
    """Find the area whose corners are closest to the point (min Euclidean)."""
    closest_area_id = None
    min_distance = float("inf")
    for area in area_data.areas.values():
        for corner in area.corners:
            corner_point = (corner.x, corner.y, corner.z)
            dist = euclidean_distance_3d(xyz, corner_point)
            if dist < min_distance:
                min_distance = dist
                closest_area_id = area.area_id
    return closest_area_id
