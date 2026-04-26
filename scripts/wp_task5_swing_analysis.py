"""Task 5: Pair kill context with actual BFS map-control swings.

For every kill, we have `before` and `after` target ticks in
`map_control_ticks.parquet` (or `map_control_ticks_sample.parquet`). The
delta in `ct_map_control` across the kill is the literal positional swing.

We correlate swing magnitude with kill_context and with round outcome.

Usage:
    uv run python scripts/wp_task5_swing_analysis.py \
        --mc-path output/full_v2/analysis/map_control_ticks_sample.parquet
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

OUT_DIR = Path("output/full_v2/analysis")


def format_markdown_table(df: pl.DataFrame, float_cols: list[str] | None = None) -> str:
    float_cols = float_cols or []
    cols = df.columns
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join([" --- "] * len(cols)) + "|"
    lines = [header, sep]
    for row in df.iter_rows():
        cells = []
        for col, val in zip(cols, row):
            if val is None:
                cells.append("")
            elif (col in float_cols or col.startswith("corr_") or col.startswith("wr_") or col.endswith("_rate") or col.endswith("_swing"))\
                and isinstance(val, (int, float)):
                if isinstance(val, float) and math.isnan(val):
                    cells.append("nan")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load(mc_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    kills = pl.read_parquet(OUT_DIR / "kills_with_context.parquet")
    mc = pl.read_parquet(mc_path)
    # Keep only the kill rows from the mc table
    return kills, mc


def build_before_after(mc: pl.DataFrame) -> pl.DataFrame:
    """For each match+round, sort by target_tick and produce (before, after)
    pairs of ct_map_control at consecutive kill events.

    'before' = last non-null ct_map_control BEFORE this kill row in the round.
    'after'  = this row's ct_map_control (the kill tick).
    """
    mc = mc.sort(["match_id", "round_num", "target_tick"])
    mc = mc.with_columns(
        [
            pl.col("ct_map_control")
            .shift(1)
            .over(["match_id", "round_num"])
            .alias("ct_mc_prev"),
            pl.col("kind").shift(1).over(["match_id", "round_num"]).alias("kind_prev"),
        ]
    )
    mc = mc.filter(pl.col("kind") == "kill").filter(pl.col("ct_mc_prev").is_not_null())
    return mc.with_columns(
        (pl.col("ct_map_control") - pl.col("ct_mc_prev")).alias("ct_mc_delta")
    )


def join_with_context(
    mc_kills: pl.DataFrame, kills: pl.DataFrame
) -> pl.DataFrame:
    # Join on (match_id, round_num, tick). mc_kills.target_tick is the kill
    # tick from the query; kills.tick is the DB kill tick. These should match.
    mc_kills = mc_kills.rename({"target_tick": "kill_tick"})
    joined = mc_kills.join(
        kills.select(
            [
                "match_id",
                "round_num",
                "tick",
                "attacker_side",
                "victim_side",
                "victim_place",
                "kill_context",
                "winner",
                "attacker_team_won",
                "is_teamkill",
                "attacker_adv_before",
                "round_time_pct",
                "map_name",
            ]
        ).rename({"tick": "kill_tick", "map_name": "map_name_k"}),
        on=["match_id", "round_num", "kill_tick"],
        how="inner",
    )
    return joined.filter(~pl.col("is_teamkill"))


def summarize_swing_by_context(df: pl.DataFrame) -> pl.DataFrame:
    # Flip delta sign for T kills so "delta" always represents swing toward
    # the attacker (positive = attacker gained control).
    df = df.with_columns(
        pl.when(pl.col("attacker_side") == "ct")
        .then(pl.col("ct_mc_delta"))
        .otherwise(-pl.col("ct_mc_delta"))
        .alias("attacker_mc_delta")
    )
    return (
        df.group_by(["attacker_side", "kill_context"])
        .agg(
            pl.len().alias("n"),
            pl.col("attacker_mc_delta").mean().alias("mean_swing"),
            pl.col("attacker_mc_delta").median().alias("median_swing"),
            pl.col("attacker_mc_delta").std().alias("std_swing"),
            pl.col("attacker_team_won").mean().alias("attacker_wr"),
        )
        .sort(["attacker_side", "kill_context"])
    )


def correlate_swing_outcome(df: pl.DataFrame) -> pl.DataFrame:
    """Correlation between mc delta and round outcome, broken out by context."""
    df = df.with_columns(
        pl.when(pl.col("attacker_side") == "ct")
        .then(pl.col("ct_mc_delta"))
        .otherwise(-pl.col("ct_mc_delta"))
        .alias("attacker_mc_delta")
    )
    rows = []
    for ctx in ["disadvantage", "parity", "advantage"]:
        sub = df.filter(pl.col("kill_context") == ctx)
        if len(sub) < 20:
            continue
        x = sub["attacker_mc_delta"].to_numpy().astype(float)
        y = sub["attacker_team_won"].to_numpy().astype(float)
        if x.std() == 0 or y.std() == 0:
            continue
        rows.append(
            {
                "kill_context": ctx,
                "n": len(sub),
                "corr_delta_win": float(np.corrcoef(x, y)[0, 1]),
                "mean_delta": float(x.mean()),
                "win_rate": float(y.mean()),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mc-path",
        type=Path,
        default=OUT_DIR / "map_control_ticks_sample.parquet",
        help="precomputed map control parquet (from wp_task4)",
    )
    args = ap.parse_args()

    kills, mc = load(args.mc_path)
    print(f"[swing] kills={len(kills):,}  mc_ticks={len(mc):,}")

    before_after = build_before_after(mc)
    print(f"[swing] before/after pairs: {len(before_after):,}")

    joined = join_with_context(before_after, kills)
    print(f"[swing] joined context rows: {len(joined):,}")

    if len(joined) == 0:
        print("[swing] no rows to analyze — did BFS tick precompute run?")
        return

    summary = summarize_swing_by_context(joined)
    print("\n## Map-control swing magnitude by kill_context x attacker_side")
    print(
        format_markdown_table(
            summary,
            float_cols=["mean_swing", "median_swing", "std_swing", "attacker_wr"],
        )
    )

    corr = correlate_swing_outcome(joined)
    print("\n## Correlation between MC swing and attacker round win")
    print(
        format_markdown_table(
            corr, float_cols=["corr_delta_win", "mean_delta", "win_rate"]
        )
    )

    md = [
        "# Task 5 — Map-control swing vs kill context",
        "",
        f"Sample size: {len(joined):,} kills with before/after BFS pairs "
        f"(from {args.mc_path.name})",
        "",
        "## Swing magnitude by context",
        format_markdown_table(
            summary,
            float_cols=["mean_swing", "median_swing", "std_swing", "attacker_wr"],
        ),
        "",
        "## Correlation: swing vs attacker_team_won",
        format_markdown_table(
            corr, float_cols=["corr_delta_win", "mean_delta", "win_rate"]
        ),
    ]
    out_md = OUT_DIR / "task5_swing_analysis.md"
    out_md.write_text("\n".join(md))
    print(f"\n[swing] wrote {out_md}")


if __name__ == "__main__":
    main()
