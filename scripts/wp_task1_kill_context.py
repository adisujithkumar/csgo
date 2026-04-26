"""Task 1: Kill-context + MC swing analysis (no BFS yet).

Reconstructs for every kill:
  - attacker_side, victim_side
  - ct_alive / t_alive BEFORE this kill (derived from prior kills in round)
  - kill_context from attacker POV: 'advantage' | 'parity' | 'disadvantage'
  - kill_sequence_in_round (1..N)
  - round outcome (winner)
  - round_time_pct (fraction of round elapsed at the kill tick)
  - map_name (joined through parsed_demos)

Writes a long-format kills-with-context parquet and prints summary tables.

Requires `output/full_v2/cs2_analytics.duckdb` — opens read-only.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import polars as pl

DB_PATH = Path("output/full_v2/cs2_analytics.duckdb")
OUT_DIR = Path("output/full_v2/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    """Open DuckDB read-only, retrying if ETL holds a write lock."""
    for attempt in range(8):
        try:
            return duckdb.connect(
                str(DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"}
            )
        except duckdb.IOException as exc:
            if attempt == 7:
                raise
            print(f"[connect] lock conflict (attempt {attempt}): {exc}")
            time.sleep(2)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

# Build per-round, per-kill state using window functions.
# Important constraints:
#   - attacker_side in kills has some NULL rows (874/108k). Exclude them.
#   - team side changes between halves: winner is stored as 'ct'/'t' at the
#     time of the round, so we can compare directly to kill.attacker_side.
#   - We compute kills_by_ct_so_far / kills_by_t_so_far using a window
#     partitioned by (match_id, round_num) ordered by (tick, index). These
#     count kills strictly BEFORE this kill is resolved.
#   - ct_alive_before_kill = 5 - kills_by_t_so_far
#   - t_alive_before_kill  = 5 - kills_by_ct_so_far
#   - After the kill is resolved, the victim_side loses one.
#
# Teamkills: if attacker_side == victim_side, still counts against victim_side.
# We capture them but tag is_teamkill so they can be filtered.

KILL_CONTEXT_SQL = """
WITH rounds_u AS (
    SELECT * FROM read_parquet(
        'output/full_v2/rounds/*.parquet',
        union_by_name = true
    )
),
kills_u AS (
    SELECT * FROM read_parquet(
        'output/full_v2/kills/*.parquet',
        union_by_name = true
    )
),
k AS (
    SELECT
        k.match_id,
        k.round_num,
        k.tick,
        k.attacker_side,
        k.victim_side,
        k.victim_place,
        k.attacker_place,
        k.weapon,
        k.headshot,
        k.distance,
        k.is_bomb_planted,
        (k.attacker_side = k.victim_side) AS is_teamkill,
        r.freeze_end AS round_freeze_end,
        r."end" AS round_end,
        r.winner,
        r.reason,
        ROW_NUMBER() OVER (
            PARTITION BY k.match_id, k.round_num
            ORDER BY k.tick, k.victim_steamid
        ) AS kill_seq,
        pd.map_name
    FROM kills_u k
    JOIN rounds_u r
      ON r.match_id = k.match_id AND r.round_num = k.round_num
    JOIN parsed_demos pd
      ON pd.match_id = k.match_id
    WHERE k.attacker_side IS NOT NULL
      AND k.victim_side IS NOT NULL
      AND r.winner IS NOT NULL
      AND r."end" > r.freeze_end
)
SELECT
    match_id,
    round_num,
    tick,
    map_name,
    attacker_side,
    victim_side,
    victim_place,
    attacker_place,
    weapon,
    headshot,
    distance,
    is_bomb_planted,
    is_teamkill,
    round_freeze_end,
    round_end,
    winner,
    reason,
    kill_seq,
    /* Counts of kills strictly BEFORE this row in the same round. */
    SUM(CASE WHEN attacker_side = 'ct' AND NOT is_teamkill THEN 1 ELSE 0 END)
        OVER (PARTITION BY match_id, round_num ORDER BY kill_seq
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        AS ct_kills_prior,
    SUM(CASE WHEN attacker_side = 't' AND NOT is_teamkill THEN 1 ELSE 0 END)
        OVER (PARTITION BY match_id, round_num ORDER BY kill_seq
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        AS t_kills_prior,
    SUM(CASE WHEN is_teamkill AND victim_side = 'ct' THEN 1 ELSE 0 END)
        OVER (PARTITION BY match_id, round_num ORDER BY kill_seq
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        AS ct_suicides_prior,
    SUM(CASE WHEN is_teamkill AND victim_side = 't' THEN 1 ELSE 0 END)
        OVER (PARTITION BY match_id, round_num ORDER BY kill_seq
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        AS t_suicides_prior
FROM k
"""


def build_kill_context(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Run the kill-context query and return a Polars DataFrame."""
    arrow = con.execute(KILL_CONTEXT_SQL).arrow()
    df = pl.from_arrow(arrow)

    # Coerce nulls from the window to zero for first-kill rows.
    df = df.with_columns(
        [
            pl.col("ct_kills_prior").fill_null(0).cast(pl.Int32),
            pl.col("t_kills_prior").fill_null(0).cast(pl.Int32),
            pl.col("ct_suicides_prior").fill_null(0).cast(pl.Int32),
            pl.col("t_suicides_prior").fill_null(0).cast(pl.Int32),
        ]
    )

    df = df.with_columns(
        [
            # alive counts just BEFORE this kill resolves
            (5 - pl.col("t_kills_prior") - pl.col("ct_suicides_prior"))
            .clip(0, 5)
            .alias("ct_alive_before"),
            (5 - pl.col("ct_kills_prior") - pl.col("t_suicides_prior"))
            .clip(0, 5)
            .alias("t_alive_before"),
        ]
    )

    # attacker man-advantage = attacker_team_alive - enemy_team_alive, computed
    # BEFORE the kill.
    df = df.with_columns(
        [
            pl.when(pl.col("attacker_side") == "ct")
            .then(pl.col("ct_alive_before") - pl.col("t_alive_before"))
            .otherwise(pl.col("t_alive_before") - pl.col("ct_alive_before"))
            .alias("attacker_adv_before"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(pl.col("attacker_adv_before") > 0)
            .then(pl.lit("advantage"))
            .when(pl.col("attacker_adv_before") == 0)
            .then(pl.lit("parity"))
            .otherwise(pl.lit("disadvantage"))
            .alias("kill_context"),
        ]
    )

    # Round time percent. Using tick units keeps the math simple; they are
    # uniform 64Hz so the ratio is identical to game_time percent.
    df = df.with_columns(
        (
            (pl.col("tick") - pl.col("round_freeze_end"))
            / pl.max_horizontal(
                pl.col("round_end") - pl.col("round_freeze_end"), pl.lit(1)
            )
        )
        .clip(0.0, 1.0)
        .alias("round_time_pct")
    )

    # Decile bin 0..9 (0 = first 10%).
    df = df.with_columns(
        ((pl.col("round_time_pct") * 10).floor().clip(0, 9).cast(pl.Int32)).alias(
            "time_decile"
        )
    )

    # attacker_won = winner == attacker_side (i.e. did the team that got the
    # kill win the round?)
    df = df.with_columns(
        (pl.col("winner") == pl.col("attacker_side")).alias("attacker_team_won")
    )

    return df


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def format_markdown_table(df: pl.DataFrame, float_cols: list[str] | None = None) -> str:
    """Render a Polars DataFrame as a GitHub markdown table."""
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
            elif col in float_cols and isinstance(val, (int, float)):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kill_context_distribution(df: pl.DataFrame) -> pl.DataFrame:
    """Counts and attacker-team win rate by kill_context."""
    clean = df.filter(~pl.col("is_teamkill"))
    grouped = (
        clean.group_by("kill_context")
        .agg(
            pl.len().alias("kills"),
            pl.col("attacker_team_won").mean().alias("attacker_wr"),
            pl.col("round_time_pct").mean().alias("avg_time_pct"),
        )
        .sort("kill_context")
    )
    return grouped


def context_by_side(df: pl.DataFrame) -> pl.DataFrame:
    """Kill context x attacker side crosstab with win rates."""
    clean = df.filter(~pl.col("is_teamkill"))
    return (
        clean.group_by(["attacker_side", "kill_context"])
        .agg(
            pl.len().alias("kills"),
            pl.col("attacker_team_won").mean().alias("attacker_wr"),
        )
        .sort(["attacker_side", "kill_context"])
    )


def top_victim_places(df: pl.DataFrame, top_n: int = 10) -> pl.DataFrame:
    """Top victim places by CT win rate given a CT kill at that place."""
    ct_kills = df.filter(
        (pl.col("attacker_side") == "ct")
        & (~pl.col("is_teamkill"))
        & pl.col("victim_place").is_not_null()
    )
    return (
        ct_kills.group_by(["map_name", "victim_place"])
        .agg(
            pl.len().alias("n_kills"),
            pl.col("attacker_team_won").mean().alias("ct_win_rate"),
            pl.col("round_time_pct").mean().alias("avg_time_pct"),
        )
        .filter(pl.col("n_kills") >= 30)
        .sort("ct_win_rate", descending=True)
        .head(top_n)
    )


def swing_by_context_and_side(df: pl.DataFrame) -> pl.DataFrame:
    """CT side win rate after a CT kill split by CT kill_context.

    This is the central question: does killing while outnumbered predict a
    better round outcome?
    """
    ct_kills = df.filter(
        (pl.col("attacker_side") == "ct") & (~pl.col("is_teamkill"))
    )
    return (
        ct_kills.group_by("kill_context")
        .agg(
            pl.len().alias("ct_kills"),
            pl.col("attacker_team_won").mean().alias("ct_win_rate"),
            pl.col("round_time_pct").mean().alias("avg_time_pct"),
            pl.col("attacker_adv_before").mean().alias("avg_adv_before"),
        )
        .sort("kill_context")
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    con = connect()
    print("[task1] running kill context query...")
    t0 = time.time()
    df = build_kill_context(con)
    print(f"[task1] built {len(df):,} rows in {time.time() - t0:.1f}s")

    out_path = OUT_DIR / "kills_with_context.parquet"
    df.write_parquet(out_path)
    print(f"[task1] wrote {out_path}")

    # Summary tables
    print("\n## Kill context distribution (attacker POV)")
    dist = kill_context_distribution(df)
    print(format_markdown_table(dist, float_cols=["attacker_wr", "avg_time_pct"]))

    print("\n## Kill context x attacker_side")
    cross = context_by_side(df)
    print(format_markdown_table(cross, float_cols=["attacker_wr"]))

    print("\n## CT swing analysis (CT kill win-rate by CT context)")
    swing = swing_by_context_and_side(df)
    print(
        format_markdown_table(
            swing, float_cols=["ct_win_rate", "avg_time_pct", "avg_adv_before"]
        )
    )

    print("\n## Top 10 victim places by CT win rate after CT kill (n >= 30)")
    top = top_victim_places(df)
    print(
        format_markdown_table(
            top, float_cols=["ct_win_rate", "avg_time_pct"]
        )
    )

    # Dump summaries to markdown file for downstream runs.
    md = [
        "# Task 1 — Kill context & outcome",
        "",
        f"Generated from {DB_PATH} ({len(df):,} non-team kills covered).",
        "",
        "## Distribution",
        format_markdown_table(dist, float_cols=["attacker_wr", "avg_time_pct"]),
        "",
        "## Context x attacker_side",
        format_markdown_table(cross, float_cols=["attacker_wr"]),
        "",
        "## CT swing analysis",
        format_markdown_table(
            swing, float_cols=["ct_win_rate", "avg_time_pct", "avg_adv_before"]
        ),
        "",
        "## Top victim_place (CT kills, n >= 30)",
        format_markdown_table(top, float_cols=["ct_win_rate", "avg_time_pct"]),
    ]
    (OUT_DIR / "task1_kill_context.md").write_text("\n".join(md))
    print(f"\n[task1] wrote summaries to {OUT_DIR / 'task1_kill_context.md'}")


if __name__ == "__main__":
    main()
