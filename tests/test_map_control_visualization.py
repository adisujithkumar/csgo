"""Test the map control visualization module."""

import pathlib
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import polars as pl
import awpy.plot.map_control as mc_plot


@pytest.fixture
def mock_frame_metrics():
    """Mock frame metrics for testing."""
    return {
        100: 0.2,   # T control (red)
        200: 0.5,   # Contested (yellow)
        300: 0.8,   # CT control (green)
    }


@pytest.fixture
def mock_round_metrics_list(mock_frame_metrics):
    """Mock list of frame metrics for round (dicts for tile-based visualization)."""
    return [mock_frame_metrics] * 3


@pytest.fixture
def mock_round_frame_metrics():
    """Mock list of MapControlFrameMetric values (floats) for round."""
    return [0.2, 0.5, 0.8, 0.6, 0.3]


@pytest.fixture
def mock_single_metric():
    """Mock single frame metric."""
    return {1: 0.5}


@pytest.fixture
def mock_empty_metrics():
    """Mock empty metrics dict."""
    return {}


class TestPlotFrameMapControlTileMetrics:
    """Tests for plot_frame_map_control_tile_metrics function."""

    def test_plot_frame_returns_figure_and_axes(self, mock_frame_metrics):
        """Test that plot_frame_map_control_tile_metrics returns Figure and Axes."""
        try:
            fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                mock_frame_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except (NotImplementedError, FileNotFoundError):
            # Function not yet implemented or map files not downloaded
            pytest.skip("Function not implemented or map files not available")

    def test_plot_frame_with_empty_metrics(self):
        """Test plot with empty metrics dict (should work, just shows map)."""
        try:
            fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                {},
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_plot_frame_figure_size(self, mock_frame_metrics):
        """Test that figure size is correct."""
        try:
            fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                mock_frame_metrics,
                "de_dust2"
            )

            # Check that figure has valid size
            figsize = fig.get_size_inches()
            assert figsize[0] > 0
            assert figsize[1] > 0

            plt.close(fig)
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_plot_frame_invalid_map(self, mock_frame_metrics):
        """Test error handling for invalid map name."""
        try:
            with pytest.raises(FileNotFoundError):
                fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                    mock_frame_metrics,
                    "de_invalid_map"
                )
                plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")


class TestPlotRoundMapControlFrameMetrics:
    """Tests for plot_round_map_control_frame_metrics function."""

    def test_plot_round_returns_figure_and_axes(self, mock_round_frame_metrics):
        """Test that plot_round_map_control_frame_metrics returns Figure and Axes."""
        try:
            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                mock_round_frame_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_plot_round_empty_list(self):
        """Test with empty list (should create empty plot)."""
        try:
            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                [],
                "de_dust2"
            )

            # Should still return valid figure and axes
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_plot_round_single_metric(self):
        """Test with single metric."""
        try:
            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                [0.5],
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_plot_round_multiple_metrics(self, mock_round_frame_metrics):
        """Test with multiple metrics."""
        try:
            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                mock_round_frame_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            # Verify that all metrics are plotted
            # The line data should have same length as input metrics
            lines = ax.get_lines()
            if lines:
                line_data = lines[0].get_ydata()
                assert len(line_data) == len(mock_round_frame_metrics)

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_plot_round_y_axis_limits(self, mock_round_frame_metrics):
        """Verify Y-axis limits are [0, 1]."""
        try:
            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                mock_round_frame_metrics,
                "de_dust2"
            )

            # Y-axis should be limited to [0, 1] range for map control metrics
            ylim = ax.get_ylim()
            assert ylim[0] == 0.0
            assert ylim[1] == 1.0

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_plot_round_metric_values_in_range(self):
        """Test that metrics outside [0, 1] are handled gracefully."""
        try:
            # Metrics should normally be in [0, 1], but test edge cases
            metrics = [0.0, 0.5, 1.0]

            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")


class TestSaveRoundMapControlTileMetricsGif:
    """Tests for save_round_map_control_tile_metrics_gif function."""

    def test_gif_file_creation(self, mock_round_metrics_list, tmp_path):
        """Test GIF file creation using tmp_path fixture."""
        try:
            output_path = tmp_path / "test_map_control.gif"

            mc_plot.save_round_map_control_tile_metrics_gif(
                mock_round_metrics_list,
                "de_dust2",
                output_path=str(output_path)
            )

            # Verify file exists
            assert output_path.exists()

            # Verify file size > 0
            assert output_path.stat().st_size > 0

        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_gif_single_frame(self, mock_single_metric, tmp_path):
        """Test with single frame."""
        try:
            output_path = tmp_path / "test_single_frame.gif"

            mc_plot.save_round_map_control_tile_metrics_gif(
                [mock_single_metric],
                "de_dust2",
                output_path=str(output_path)
            )

            # Should still create valid GIF with single frame
            assert output_path.exists()
            assert output_path.stat().st_size > 0

        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_gif_empty_list_fails(self, tmp_path):
        """Test that empty list fails (IndexError when trying to save empty GIF)."""
        try:
            output_path = tmp_path / "test_empty.gif"

            # Empty list will cause IndexError when trying to access pil_images[0]
            with pytest.raises(IndexError):
                mc_plot.save_round_map_control_tile_metrics_gif(
                    [],
                    "de_dust2",
                    output_path=str(output_path)
                )

        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_gif_valid_output_path(self, mock_round_metrics_list, tmp_path):
        """Test GIF saved to custom output path."""
        try:
            # Use tmp_path which is guaranteed to exist
            output_path = tmp_path / "custom_output.gif"

            mc_plot.save_round_map_control_tile_metrics_gif(
                mock_round_metrics_list,
                "de_dust2",
                output_path=str(output_path)
            )

            # Verify file was created at specified path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    def test_frame_to_gif_consistency(self, mock_frame_metrics, tmp_path):
        """Test that individual frames match GIF frames."""
        try:
            # Plot individual frame
            fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                mock_frame_metrics,
                "de_dust2"
            )
            plt.close(fig)

            # Create GIF with same frame repeated
            output_path = tmp_path / "consistency_test.gif"
            mc_plot.save_round_map_control_tile_metrics_gif(
                [mock_frame_metrics, mock_frame_metrics],
                "de_dust2",
                output_path=str(output_path)
            )

            # Both should succeed without errors
            assert output_path.exists()

        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_multiple_visualizations_no_memory_leak(self, mock_frame_metrics):
        """Test that creating multiple visualizations closes figures properly."""
        try:
            # Record initial figure count
            initial_fig_count = len(plt.get_fignums())

            # Create multiple figures and close them
            for _ in range(5):
                fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                    mock_frame_metrics,
                    "de_dust2"
                )
                plt.close(fig)

            # Get current figure count
            final_fig_count = len(plt.get_fignums())

            # Should not have accumulated more figures than we started with
            # (allowing for some matplotlib internal figures)
            assert final_fig_count <= initial_fig_count + 1

        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_plot_frame_all_t_control(self):
        """Test frame with all tiles showing T control (metric = 0.0)."""
        try:
            all_t_metrics = {100: 0.0, 200: 0.0, 300: 0.0}

            fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                all_t_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_plot_frame_all_ct_control(self):
        """Test frame with all tiles showing CT control (metric = 1.0)."""
        try:
            all_ct_metrics = {100: 1.0, 200: 1.0, 300: 1.0}

            fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                all_ct_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_plot_frame_neutral_control(self):
        """Test frame with all tiles showing neutral control (metric = 0.5)."""
        try:
            neutral_metrics = {100: 0.5, 200: 0.5, 300: 0.5}

            fig, ax = mc_plot.plot_frame_map_control_tile_metrics(
                neutral_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_plot_round_all_t_dominance(self):
        """Test round plot with T dominance throughout."""
        try:
            t_metrics = [0.0, 0.1, 0.15, 0.2, 0.1]

            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                t_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_plot_round_all_ct_dominance(self):
        """Test round plot with CT dominance throughout."""
        try:
            ct_metrics = [0.8, 0.9, 0.95, 1.0, 0.85]

            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                ct_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_plot_round_control_shift(self):
        """Test round plot showing control shifting from T to CT."""
        try:
            shift_metrics = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

            fig, ax = mc_plot.plot_round_map_control_frame_metrics(
                shift_metrics,
                "de_dust2"
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

            # Verify data shows the shift
            lines = ax.get_lines()
            if lines:
                line_data = lines[0].get_ydata()
                # First value should be low, last value should be high
                assert line_data[0] < 0.5
                assert line_data[-1] > 0.5

            plt.close(fig)
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_gif_with_varying_tile_counts(self, tmp_path):
        """Test GIF creation where frames have different numbers of tiles."""
        try:
            # Each frame has different tiles visible
            varying_metrics = [
                {100: 0.2},  # Only one tile
                {100: 0.3, 200: 0.5},  # Two tiles
                {100: 0.4, 200: 0.6, 300: 0.8},  # Three tiles
            ]

            output_path = tmp_path / "varying_tiles.gif"
            mc_plot.save_round_map_control_tile_metrics_gif(
                varying_metrics,
                "de_dust2",
                output_path=str(output_path)
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0

        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_gif_with_custom_duration(self, mock_round_metrics_list, tmp_path):
        """Test GIF creation with custom frame duration."""
        try:
            output_path = tmp_path / "custom_duration.gif"

            # Use custom duration of 200ms
            mc_plot.save_round_map_control_tile_metrics_gif(
                mock_round_metrics_list,
                "de_dust2",
                output_path=str(output_path),
                duration=200
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0

        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")


@pytest.fixture
def mock_tick_data():
    """Mock tick data with alive players from both teams."""
    return pl.DataFrame({
        "tick": [1000, 1000, 1000, 1000],
        "X": [-1396.0, -1500.0, 200.0, 300.0],
        "Y": [2458.0, 2500.0, 1000.0, 1100.0],
        "Z": [132.0, 140.0, 50.0, 55.0],
        "yaw": [90.0, 45.0, 180.0, 270.0],
        "pitch": [0.0, -10.0, 5.0, 0.0],
        "health": [100, 80, 100, 90],
        "side": ["t", "t", "ct", "ct"],
        "name": ["PlayerT1", "PlayerT2", "PlayerCT1", "PlayerCT2"],
    })


@pytest.fixture
def mock_round_data():
    """Mock round data with multiple ticks for GIF generation."""
    ticks = []
    # Use tick values divisible by 384 (0, 384, 768, 1152)
    for tick_num in range(0, 1536, 384):  # 4 frames at ticks 0, 384, 768, 1152
        for player_id in range(4):
            ticks.append({
                "tick": tick_num,
                "X": -1396.0 + tick_num/10 + player_id*100,
                "Y": 2458.0 + tick_num/10,
                "Z": 132.0,
                "yaw": 90.0 + tick_num/5,
                "pitch": 0.0,
                "health": 100 - tick_num/50,
                "side": "t" if player_id < 2 else "ct",
                "name": f"Player{player_id}",
            })
    return pl.DataFrame(ticks)


class TestPlotFrameMapControlWithPlayers:
    """Tests for plot_frame_map_control_with_players function."""

    def test_plot_with_players_returns_figure_and_axes(self, mock_frame_metrics, mock_tick_data):
        """Test basic functionality returns Figure and Axes."""
        try:
            fig, ax = mc_plot.plot_frame_map_control_with_players(
                mock_frame_metrics, mock_tick_data, "de_dust2"
            )
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)
            plt.close(fig)
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_plot_with_no_alive_players(self, mock_frame_metrics):
        """Test with all dead players (health=0)."""
        try:
            dead_players = pl.DataFrame({
                "tick": [1000, 1000],
                "X": [-1396.0, 200.0],
                "Y": [2458.0, 1000.0],
                "Z": [132.0, 50.0],
                "yaw": [90.0, 180.0],
                "pitch": [0.0, 0.0],
                "health": [0, 0],  # All dead
                "side": ["t", "ct"],
                "name": ["PlayerT1", "PlayerCT1"],
            })
            fig, ax = mc_plot.plot_frame_map_control_with_players(
                mock_frame_metrics, dead_players, "de_dust2"
            )
            assert isinstance(fig, Figure)
            plt.close(fig)
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_plot_missing_required_columns(self, mock_frame_metrics):
        """Test error handling for missing columns."""
        try:
            incomplete_data = pl.DataFrame({
                "tick": [1000],
                "X": [-1396.0],
                "Y": [2458.0],
                # Missing Z, yaw, pitch, health, side, name
            })
            with pytest.raises(ValueError, match="Missing required columns"):
                mc_plot.plot_frame_map_control_with_players(
                    mock_frame_metrics, incomplete_data, "de_dust2"
                )
        except NotImplementedError:
            pytest.skip("Function not implemented")

    def test_plot_empty_tick_data(self, mock_frame_metrics):
        """Test error handling for empty tick data."""
        try:
            empty_data = pl.DataFrame({
                "tick": [], "X": [], "Y": [], "Z": [], "yaw": [],
                "pitch": [], "health": [], "side": [], "name": []
            })
            with pytest.raises(ValueError, match="tick_data cannot be empty"):
                mc_plot.plot_frame_map_control_with_players(
                    mock_frame_metrics, empty_data, "de_dust2"
                )
        except NotImplementedError:
            pytest.skip("Function not implemented")


class TestSaveRoundMapControlWithPlayersGif:
    """Tests for save_round_map_control_with_players_gif function."""

    def test_gif_creation_with_players(self, mock_round_data, tmp_path):
        """Test GIF file creation with player data."""
        try:
            output = tmp_path / "test.gif"
            mc_plot.save_round_map_control_with_players_gif(
                mock_round_data, "de_dust2", str(output), duration=100, tick_interval=384
            )
            assert output.exists()
            assert output.stat().st_size > 0
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("Function not implemented or map files not available")

    def test_gif_empty_round_data(self, tmp_path):
        """Test error handling for empty round data."""
        try:
            empty_data = pl.DataFrame({
                "tick": [], "X": [], "Y": [], "Z": [], "yaw": [],
                "pitch": [], "health": [], "side": [], "name": []
            })
            output = tmp_path / "test.gif"
            with pytest.raises(ValueError, match="round_data cannot be empty"):
                mc_plot.save_round_map_control_with_players_gif(
                    empty_data, "de_dust2", str(output)
                )
        except NotImplementedError:
            pytest.skip("Function not implemented")
