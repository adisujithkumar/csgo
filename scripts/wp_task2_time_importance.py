"""Task 2: Round-time importance + victim-place positional proxy.

Answers:
  * In which decile of the round does `ct_alive - t_alive` (and other
    easy features) best predict the round winner? This is the proxy for
    "when does map control matter most" before real BFS is plugged in.
  * Does a CT kill at a "disadvantage" actually produce an outsized
    positional swing?  Uses `victim_place` as the positional proxy.

Writes `output/full_v2/analysis/task2_time_importance.md` and two parquet
artifacts:
  * `alive_diff_by_decile.parquet`
  * `place_value_proxy.parquet`
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

DB_PATH = Path("output/full_v2/cs2_analytics.duckdb")
OUT_DIR = Path("output/full_v2/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    """Open DuckDB read-only, retrying if the ETL holds the write lock."""
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
# SQL helpers
# ---------------------------------------------------------------------------

TICK_SAMPLE_SQL = """
WITH rounds_u AS (
    SELECT * FROM read_parquet(
        'output/full_v2/rounds/*.parquet', union_by_name = true
    )
),
ticks_u AS (
    SELECT * FROM read_parquet(
        'output/full_v2/ticks/*.parquet', union_by_name = true
    )
),
/* For each round, pick a set of evenly spaced ticks (every 12 sampled rows).
 * At 16-tick sampling that is ~3 seconds between rows.
 */
agg AS (
    SELECT
        t.match_id,
        t.round_num,
        t.tick,
        SUM(CASE WHEN t.side = 'ct' AND t.health > 0 THEN 1 ELSE 0 END) AS ct_alive,
        SUM(CASE WHEN t.side = 't'  AND t.health > 0 THEN 1 ELSE 0 END) AS t_alive,
        SUM(CASE WHEN t.side = 'ct' AND t.health > 0 THEN t.health ELSE 0 END) AS hp_ct,
        SUM(CASE WHEN t.side = 't'  AND t.health > 0 THEN t.health ELSE 0 END) AS hp_t,
        SUM(CASE WHEN t.side = 'ct' THEN t.current_equip_value ELSE 0 END) AS equip_ct,
        SUM(CASE WHEN t.side = 't'  THEN t.current_equip_value ELSE 0 END) AS equip_t,
        MAX(CASE WHEN t.is_bomb_planted THEN 1 ELSE 0 END) AS is_bomb_planted
    FROM ticks_u t
    GROUP BY t.match_id, t.round_num, t.tick
)
SELECT
    a.match_id,
    a.round_num,
    a.tick,
    r.freeze_end,
    r."end" AS round_end,
    r.winner,
    pd.map_name,
    a.ct_alive,
    a.t_alive,
    a.hp_ct,
    a.hp_t,
    a.equip_ct,
    a.equip_t,
    a.is_bomb_planted,
    (a.tick - r.freeze_end) * 1.0
        / NULLIF(r."end" - r.freeze_end, 0) AS round_time_pct
FROM agg a
JOIN rounds_u r
  ON r.match_id = a.match_id AND r.round_num = a.round_num
JOIN parsed_demos pd
  ON pd.match_id = a.match_id
WHERE a.tick >= r.freeze_end
  AND a.tick <= r."end"
  AND r.winner IS NOT NULL
  AND r."end" > r.freeze_end
  /* downsample: keep every 12th distinct tick per round (~3s grid) */
  AND ((a.tick - r.freeze_end) / 16) % 12 = 0
"""


def build_tick_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    arrow = con.execute(TICK_SAMPLE_SQL).arrow()
    df = pl.from_arrow(arrow)
    df = df.with_columns(
        [
            pl.col("round_time_pct").clip(0.0, 1.0).alias("round_time_pct"),
            (pl.col("winner") == "ct").cast(pl.Int8).alias("ct_won"),
        ]
    )
    df = df.with_columns(
        ((pl.col("round_time_pct") * 10).floor().clip(0, 9).cast(pl.Int32)).alias(
            "time_decile"
        )
    )
    df = df.with_columns(
        [
            (pl.col("ct_alive") - pl.col("t_alive")).alias("alive_diff"),
            (pl.col("hp_ct") - pl.col("hp_t")).alias("hp_diff"),
            (pl.col("equip_ct") - pl.col("equip_t")).alias("equip_diff"),
        ]
    )
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def pointbiserial_decile(df: pl.DataFrame) -> pl.DataFrame:
    """Correlation between features and ct_won within each decile."""
    results: list[dict] = []
    for decile in range(10):
        sub = df.filter(pl.col("time_decile") == decile)
        if len(sub) < 500:
            continue
        row = {"time_decile": decile, "n_ticks": len(sub)}
        for feat in ["alive_diff", "hp_diff", "equip_diff", "is_bomb_planted"]:
            x = sub[feat].to_numpy().astype(float)
            y = sub["ct_won"].to_numpy().astype(float)
            if x.std() == 0 or y.std() == 0:
                row[f"corr_{feat}"] = float("nan")
                continue
            row[f"corr_{feat}"] = float(np.corrcoef(x, y)[0, 1])
        results.append(row)
    return pl.DataFrame(results)


def logistic_coef_per_decile(df: pl.DataFrame) -> pl.DataFrame:
    """Fit a 1-D logistic regression of ct_won ~ alive_diff per decile."""
    from sklearn.linear_model import LogisticRegression

    rows: list[dict] = []
    for decile in range(10):
        sub = df.filter(pl.col("time_decile") == decile)
        if len(sub) < 500:
            continue
        X = sub[["alive_diff"]].to_numpy()
        y = sub["ct_won"].to_numpy()
        if y.mean() in (0.0, 1.0):
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        auc_proxy = float(
            ((model.predict_proba(X)[:, 1] - y) ** 2).mean() ** 0.5
        )  # RMSE brier proxy
        rows.append(
            {
                "time_decile": decile,
                "n_ticks": len(sub),
                "coef_alive_diff": float(model.coef_[0][0]),
                "intercept": float(model.intercept_[0]),
                "brier_rmse": auc_proxy,
            }
        )
    return pl.DataFrame(rows)


def top_place_values(kill_df: pl.DataFrame, side: str, top_n: int = 20) -> pl.DataFrame:
    clean = kill_df.filter(
        (pl.col("attacker_side") == side)
        & (~pl.col("is_teamkill"))
        & pl.col("victim_place").is_not_null()
    )
    won_col = "attacker_team_won"
    return (
        clean.group_by(["map_name", "victim_place"])
        .agg(
            pl.len().alias("n_kills"),
            pl.col(won_col).mean().alias(f"{side}_win_rate_given_kill"),
            pl.col("round_time_pct").mean().alias("avg_time_pct"),
            (
                pl.col("kill_context").eq("disadvantage").cast(pl.Float64).mean()
            ).alias("pct_disadv"),
        )
        .filter(pl.col("n_kills") >= 40)
        .sort(f"{side}_win_rate_given_kill", descending=True)
        .head(top_n)
    )


def place_swing_proxy(kill_df: pl.DataFrame) -> pl.DataFrame:
    """For each (map, victim_place), compute the difference in round win rate
    when the attacker was disadvantaged vs parity vs advantaged. The bigger
    the disadvantage-vs-parity gap, the bigger the implied positional swing
    at that place (because killing there under pressure still flips rounds).
    """
    clean = kill_df.filter(
        (pl.col("attacker_side") == "ct") & ~pl.col("is_teamkill")
    )
    agg = (
        clean.group_by(["map_name", "victim_place", "kill_context"])
        .agg(
            pl.len().alias("n"),
            pl.col("attacker_team_won").mean().alias("wr"),
        )
        .sort(["map_name", "victim_place", "kill_context"])
    )
    pivot = agg.pivot(
        values=["n", "wr"],
        index=["map_name", "victim_place"],
        on="kill_context",
        aggregate_function="first",
    ).fill_null(0)

    # Polars pivot with multiple values produces columns like n_advantage, wr_advantage, etc.
    required = ["wr_advantage", "wr_parity", "wr_disadvantage", "n_disadvantage"]
    missing = [c for c in required if c not in pivot.columns]
    if missing:
        return pivot  # defensive
    return (
        pivot.with_columns(
            (pl.col("wr_disadvantage") - pl.col("wr_parity")).alias(
                "disadv_minus_parity"
            )
        )
        .filter(pl.col("n_disadvantage") >= 20)
        .sort("disadv_minus_parity", descending=True)
        .head(15)
    )


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
            elif (col in float_cols or col.startswith("corr_") or col.startswith("wr_")) and isinstance(
                val, (int, float)
            ):
                if isinstance(val, float) and math.isnan(val):
                    cells.append("nan")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    con = connect()
    print("[task2] sampling ticks...")
    t0 = time.time()
    tick_df = build_tick_features(con)
    print(f"[task2] built {len(tick_df):,} sampled rows in {time.time() - t0:.1f}s")
    tick_df.write_parquet(OUT_DIR / "alive_diff_samples.parquet")

    print("\n## Decile-wise correlation with CT round win")
    corr = pointbiserial_decile(tick_df)
    print(
        format_markdown_table(
            corr,
            float_cols=["corr_alive_diff", "corr_hp_diff", "corr_equip_diff", "corr_is_bomb_planted"],
        )
    )

    print("\n## Logistic regression of ct_won ~ alive_diff per decile")
    logit = logistic_coef_per_decile(tick_df)
    print(
        format_markdown_table(
            logit, float_cols=["coef_alive_diff", "intercept", "brier_rmse"]
        )
    )

    kill_df = pl.read_parquet(OUT_DIR / "kills_with_context.parquet")

    print("\n## Top victim_place by CT win rate after CT kill")
    ct_top = top_place_values(kill_df, "ct")
    print(
        format_markdown_table(
            ct_top,
            float_cols=["ct_win_rate_given_kill", "avg_time_pct", "pct_disadv"],
        )
    )

    print("\n## Top victim_place by T win rate after T kill")
    t_top = top_place_values(kill_df, "t")
    print(
        format_markdown_table(
            t_top,
            float_cols=["t_win_rate_given_kill", "avg_time_pct", "pct_disadv"],
        )
    )

    print("\n## CT kills where disadvantaged kills flip rounds most (vs parity baseline)")
    swing = place_swing_proxy(kill_df)
    print(format_markdown_table(swing))
    swing.write_parquet(OUT_DIR / "place_value_proxy.parquet")

    md = [
        "# Task 2 — round-time importance + place-value proxy",
        "",
        f"Sampled tick rows used: {len(tick_df):,}",
        "",
        "## Decile-wise correlation with CT round win",
        format_markdown_table(
            corr,
            float_cols=["corr_alive_diff", "corr_hp_diff", "corr_equip_diff", "corr_is_bomb_planted"],
        ),
        "",
        "## Logistic regression of ct_won ~ alive_diff per decile",
        format_markdown_table(
            logit,
            float_cols=["coef_alive_diff", "intercept", "brier_rmse"],
        ),
        "",
        "## Top victim_place — CT kills",
        format_markdown_table(
            ct_top,
            float_cols=["ct_win_rate_given_kill", "avg_time_pct", "pct_disadv"],
        ),
        "",
        "## Top victim_place — T kills",
        format_markdown_table(
            t_top,
            float_cols=["t_win_rate_given_kill", "avg_time_pct", "pct_disadv"],
        ),
        "",
        "## CT disadvantage-kill swing",
        format_markdown_table(swing),
    ]
    (OUT_DIR / "task2_time_importance.md").write_text("\n".join(md))
    print(f"\n[task2] wrote {OUT_DIR / 'task2_time_importance.md'}")


if __name__ == "__main__":
    main()
