from dataclasses import dataclass
from typing import Literal, NotRequired, TypeAlias, TypeGuard, final, overload
from typing_extensions import TypedDict
import polars as pl

class PlayerInfo(TypedDict):
    """PlayerInfo at time t."""
    x: float
    y: float
    z: float

    """
    steamID: int
    name: str
    team: str
    side: str
    eyeX: float
    eyeY: float
    eyeZ: float
    velocityX: float
    velocityY: float
    velocityZ: float
    viewX: float
    viewY: float
    hp: int
    armor: int
    activeWeapon: str
    flashGrenades: int
    smokeGrenades: int
    heGrenades: int
    fireGrenades: int
    totalUtility: int
    lastPlaceName: str
    isAlive: bool
    isBot: bool
    isBlinded: bool
    isAirborne: bool
    isDucking: bool
    isDuckingInProgress: bool
    isUnDuckingInProgress: bool
    isDefusing: bool
    isPlanting: bool
    isReloading: bool
    isInBombZone: bool
    isInBuyZone: bool
    isStanding: bool
    isScoped: bool
    isWalking: bool
    isUnknown: bool
    inventory: list[WeaponInfo] | None
    spotters: list[int] | None
    equipmentValue: int
    equipmentValueFreezetimeEnd: int
    equipmentValueRoundStart: int
    cash: int
    cashSpendThisRound: int
    cashSpendTotal: int
    hasHelmet: bool
    hasDefuse: bool
    hasBomb: bool
    ping: int
    zoomLevel: int
    """

class TeamFrameInfo(TypedDict):
    """TeamFrameInfo at time t."""

    side: str
    """
    teamName: str
    teamEqVal: int
    """
    alivePlayers: int
    """
    totalUtility: int
    """
    players: list[PlayerInfo] | None

# Type to represent player position (list of list (x, y, z))
PlayerPosition: TypeAlias = list[float, float, float]

@dataclass
class TeamMetadata:
    """Dataclass containing metadata for one team.

    Holds information for alive player locations. Can include
    more metadata (utility, bomb location, etc.) in the future
    """

    alive_player_locations: list[PlayerPosition]

@dataclass
class FrameTeamMetadata:
    """Dataclass with metadata on both teams in frame.

    Return type for awpy.analytics.map_control.extract_teams_metadata.
    Holds parsed metadata object (TeamMetadata) for both teams
    """

    t_metadata: TeamMetadata
    ct_metadata: TeamMetadata


def _extract_team_metadata(
    side_data: pl.DataFrame,  # Ensure type hint is correct
) -> TeamMetadata:
    """Helper function to parse player locations in given side_data.

    Args:
        side_data (pl.DataFrame): Polars DataFrame with metadata for side's players.

    Returns:
        TeamMetadata with metadata on team's players
    """

    # Extract alive player locations
    if side_data.is_empty():
        return TeamMetadata(alive_player_locations=[])

    alive_players = side_data[['X', 'Y', 'Z']].to_numpy().tolist()
    return TeamMetadata(alive_player_locations=alive_players)

def extract_teams_metadata(
    frame: pl.DataFrame,
) -> FrameTeamMetadata:
    """Parse frame data for alive player locations for both sides.

    Args:
        frame (GameFrame): Dictionary in the form of an awpy frame
            containing relevant data for both sides

    Returns:
        FrameTeamMetadata containing team metadata (player
            positions, etc.)
    """
    return FrameTeamMetadata(
        t_metadata=_extract_team_metadata(frame.filter(pl.col("side") == 't')),
        ct_metadata=_extract_team_metadata(frame.filter(pl.col("side") == 'ct')),
    )
