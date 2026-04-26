"""
visualize_wp_curves.py
======================
Plots win-probability curves for 4 "interesting" rounds selected automatically
from OOF predictions, annotated with kill events from DuckDB.

Output: docs/examples/wp_curves.png
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OOF_PARQUET = REPO_ROOT / "output" / "poc_dust2" / "wp_model" / "oof_predictions.parquet"
DUCKDB_PATH = REPO_ROOT / "output" / "full_dust2" / "cs2_analytics.duckdb"
OUT_PNG = REPO_ROOT / "docs" / "examples" / "wp_curves.png"
MAX_ROUND_SECONDS = 115.0  # regulation round length (freeze-end → end)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_oof() -> pl.DataFrame:
    df = pl.read_parquet(OOF_PARQUET)
    return df


def connect_db(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def get_overlapping_match_ids(oof: pl.DataFrame, con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return match_ids present in both OOF parquet and the kills table."""
    oof_ids = oof["match_id"].unique().to_list()
    rows = con.execute(
        f"SELECT DISTINCT match_id FROM kills WHERE match_id IN {tuple(oof_ids)}"
    ).fetchall()
    return [r[0] for r in rows]


def get_round_tick_range(
    match_id: str, round_num: int, con: duckdb.DuckDBPyConnection
) -> tuple[int, int] | None:
    """Return (freeze_end, end) ticks for the given round. None if not found."""
    row = con.execute(
        'SELECT freeze_end, "end" FROM rounds WHERE match_id = ? AND round_num = ?',
        [match_id, round_num],
    ).fetchone()
    return row if row else None


def get_kills(
    match_id: str, round_num: int, con: duckdb.DuckDBPyConnection
) -> pl.DataFrame:
    """Fetch kills for a specific round."""
    df = con.execute(
        """
        SELECT tick, attacker_side, victim_side, weapon
        FROM kills
        WHERE match_id = ? AND round_num = ?
        ORDER BY tick
        """,
        [match_id, round_num],
    ).pl()
    return df


def tick_to_pct(tick: int, freeze_end: int, end_tick: int) -> float:
    span = end_tick - freeze_end
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (tick - freeze_end) / span))


# ---------------------------------------------------------------------------
# Round selection
# ---------------------------------------------------------------------------

def select_interesting_rounds(
    oof: pl.DataFrame, overlap_ids: list[str], con: duckdb.DuckDBPyConnection
) -> list[dict]:
    """
    Pick 4 rounds:
      1. Biggest comeback  (pred_mc swings ≥ 0.4 crossing 0.5 in both directions)
      2. Clean CT win      (early pred_mc > 0.7, ct_won = 1)
      3. Clean T win       (early pred_mc < 0.3, ct_won = 0)
      4. Highest variance  (std of pred_mc)
    Each dict: {match_id, round_num, label}
    """
    filtered = oof.filter(pl.col("match_id").is_in(overlap_ids))

    per_round = filtered.group_by(["match_id", "round_num"]).agg(
        [
            pl.col("pred_mc").min().alias("mc_min"),
            pl.col("pred_mc").max().alias("mc_max"),
            pl.col("pred_mc").std().alias("mc_std"),
            pl.col("ct_won").first().alias("ct_won"),
        ]
    ).with_columns((pl.col("mc_max") - pl.col("mc_min")).alias("mc_range"))

    # ---- 1. Comeback -------------------------------------------------------
    # Requires crossing 0.5 in both directions AND large swing
    comeback_cands = per_round.filter(
        (pl.col("mc_min") < 0.3) & (pl.col("mc_max") > 0.7)
    ).sort("mc_range", descending=True)

    comeback = _pick_first_with_kills(comeback_cands, con, "Biggest comeback")

    # ---- 2. Clean CT win ---------------------------------------------------
    early = filtered.filter(pl.col("time_elapsed_pct") < 0.3).group_by(
        ["match_id", "round_num"]
    ).agg(
        pl.col("pred_mc").mean().alias("early_mc"),
        pl.col("ct_won").first().alias("ct_won"),
    )
    ct_cands = early.filter(
        (pl.col("ct_won") == 1) & (pl.col("early_mc") > 0.65)
    ).sort("early_mc", descending=True)

    ct_win = _pick_first_with_kills(ct_cands, con, "Clean CT win")

    # ---- 3. Clean T win ----------------------------------------------------
    t_cands = early.filter(
        (pl.col("ct_won") == 0) & (pl.col("early_mc") < 0.35)
    ).sort("early_mc")

    t_win = _pick_first_with_kills(t_cands, con, "Clean T win")

    # ---- 4. Highest variance -----------------------------------------------
    # Exclude already selected rounds
    already = {
        (d["match_id"], d["round_num"]) for d in [comeback, ct_win, t_win] if d
    }
    var_cands = (
        per_round.filter(pl.col("mc_std").is_not_null())
        .sort("mc_std", descending=True)
    )
    high_var = None
    for row in var_cands.iter_rows(named=True):
        if (row["match_id"], row["round_num"]) in already:
            continue
        kills = get_kills(row["match_id"], row["round_num"], con)
        if len(kills) >= 3:
            high_var = {
                "match_id": row["match_id"],
                "round_num": row["round_num"],
                "label": "Highest WP variance",
            }
            break

    result = [r for r in [comeback, ct_win, t_win, high_var] if r is not None]

    if len(result) < 4:
        print(f"Warning: only {len(result)}/4 interesting rounds found.", file=sys.stderr)
    return result[:4]


def _pick_first_with_kills(
    cands: pl.DataFrame, con: duckdb.DuckDBPyConnection, label: str
) -> dict | None:
    for row in cands.iter_rows(named=True):
        kills = get_kills(row["match_id"], row["round_num"], con)
        if len(kills) >= 1:
            return {
                "match_id": row["match_id"],
                "round_num": row["round_num"],
                "label": label,
            }
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CT_COLOR = "#4a90d9"   # blue
T_COLOR = "#e05c4b"    # red
CT_SHADE = "#cce4f7"
T_SHADE = "#fad4cf"
GRID_COLOR = "#e0e0e0"


def plot_round(
    ax: plt.Axes,
    round_info: dict,
    oof: pl.DataFrame,
    con: duckdb.DuckDBPyConnection,
) -> None:
    match_id = round_info["match_id"]
    round_num = round_info["round_num"]
    label = round_info["label"]

    # OOF predictions for this round
    rdf = (
        oof.filter(
            (pl.col("match_id") == match_id) & (pl.col("round_num") == round_num)
        )
        .sort("time_elapsed_pct")
    )

    x = rdf["time_elapsed_pct"].to_numpy()
    y_bl = rdf["pred_baseline"].to_numpy()
    y_mc = rdf["pred_mc"].to_numpy()
    ct_won = int(rdf["ct_won"][0])
    outcome = "CT won" if ct_won == 1 else "T won"

    # --- Background shading -------------------------------------------------
    ax.fill_between(x, 0.5, y_mc, where=(y_mc >= 0.5), color=CT_SHADE, alpha=0.55, zorder=1)
    ax.fill_between(x, 0.5, y_mc, where=(y_mc < 0.5), color=T_SHADE, alpha=0.55, zorder=1)

    # --- WP curves ----------------------------------------------------------
    ax.plot(x, y_bl, color="gray", lw=1.4, ls="--", alpha=0.7, label="Baseline", zorder=3)
    ax.plot(x, y_mc, color="#1a1a2e", lw=2.0, ls="-", label="Map-control", zorder=4)

    # --- 0.5 reference line -------------------------------------------------
    ax.axhline(0.5, color="#555555", lw=0.8, ls=":", zorder=2)

    # --- Kill annotations ---------------------------------------------------
    kills = get_kills(match_id, round_num, con)
    tick_range = get_round_tick_range(match_id, round_num, con)

    if tick_range and len(kills) > 0:
        freeze_end, end_tick = tick_range
        for row in kills.iter_rows(named=True):
            pct = tick_to_pct(row["tick"], freeze_end, end_tick)
            victim_side = (row["victim_side"] or "").lower()
            color = CT_COLOR if victim_side == "ct" else T_COLOR
            label_txt = "CT−" if victim_side == "ct" else "T−"
            ax.axvline(pct, color=color, lw=1.0, alpha=0.8, ymin=0.0, ymax=0.12, zorder=5)
            # Small label near bottom
            ax.text(
                pct, 0.03, label_txt,
                ha="center", va="bottom", fontsize=5.5, color=color,
                rotation=90, zorder=6,
            )

    # --- X-axis: convert pct → seconds ------------------------------------
    def pct_to_sec(p):
        return p * MAX_ROUND_SECONDS

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    tick_pcts = np.linspace(0, 1, 7)  # 0, ~19, ~38, … 115
    ax.set_xticks(tick_pcts)
    ax.set_xticklabels([f"{pct_to_sec(p):.0f}s" for p in tick_pcts], fontsize=8)

    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)

    ax.set_xlabel("Round time", fontsize=9)
    ax.set_ylabel("CT win probability", fontsize=9)

    ax.set_title(
        f"Round {round_num} — {outcome}\n({label})",
        fontsize=10, fontweight="bold", pad=6,
    )
    ax.grid(True, color=GRID_COLOR, lw=0.6, zorder=0)
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")


def build_legend(fig: plt.Figure) -> None:
    handles = [
        mpatches.Patch(facecolor=CT_SHADE, edgecolor=CT_COLOR, label="CT advantage"),
        mpatches.Patch(facecolor=T_SHADE, edgecolor=T_COLOR, label="T advantage"),
        plt.Line2D([0], [0], color="#1a1a2e", lw=2, ls="-", label="Map-control WP"),
        plt.Line2D([0], [0], color="gray", lw=1.4, ls="--", label="Baseline WP"),
        plt.Line2D([0], [0], color=CT_COLOR, lw=2, label="CT player killed"),
        plt.Line2D([0], [0], color=T_COLOR, lw=2, label="T player killed"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.01),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading OOF predictions …")
    oof = load_oof()
    print(f"  {oof.shape[0]:,} rows, {oof['match_id'].n_unique()} unique matches")

    print("Connecting to DuckDB …")
    con = connect_db()

    print("Finding overlapping match IDs …")
    overlap_ids = get_overlapping_match_ids(oof, con)
    print(f"  {len(overlap_ids)} match IDs found in both datasets")

    print("Selecting interesting rounds …")
    rounds = select_interesting_rounds(oof, overlap_ids, con)
    for r in rounds:
        print(f"  [{r['label']}] match={r['match_id']}  round={r['round_num']}")

    if not rounds:
        print("ERROR: No rounds selected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Pad to 4 if fewer found
    while len(rounds) < 4:
        rounds.append(rounds[-1])  # repeat last as fallback

    # --- Build figure -------------------------------------------------------
    print("Plotting …")
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.07, right=0.97,
        top=0.93, bottom=0.12,
        hspace=0.45, wspace=0.28,
    )

    fig.suptitle(
        "Win Probability Curves — CS2 Dust2 (Map-Control Model vs Baseline)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for ax, round_info in zip(axes.flat, rounds):
        plot_round(ax, round_info, oof, con)

    build_legend(fig)

    # --- Save ---------------------------------------------------------------
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
