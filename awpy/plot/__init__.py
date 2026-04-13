"""Awpy plotting module."""

PLOT_SETTINGS = {
    "ct": {
        "marker": "o",
        "color": "tab:cyan",
        "size": 8,
    },
    "t": {
        "marker": "o",
        "color": "tab:olive",
        "size": 8,
    },
    "bomb": {
        "marker": "x",
        "color": "tab:orange",
        "size": 8,
    },
    "smoke": {
        "marker": "o",
        "color": "tab:gray",
        "size": 12,
    },
    "fire": {
        "marker": "o",
        "color": "tab:red",
        "size": 12,
    },
}

from awpy.plot.plot import gif, heatmap, plot
from awpy.plot.map_control import (
    plot_frame_map_control_tile_metrics,
    plot_round_map_control_frame_metrics,
    save_round_map_control_tile_metrics_gif,
    plot_frame_map_control_with_players,
    save_round_map_control_with_players_gif,
)

__all__ = [
    "gif",
    "heatmap",
    "plot",
    "plot_frame_map_control_tile_metrics",
    "plot_round_map_control_frame_metrics",
    "save_round_map_control_tile_metrics_gif",
    "plot_frame_map_control_with_players",
    "save_round_map_control_with_players_gif",
]
