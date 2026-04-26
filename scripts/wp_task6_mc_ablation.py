"""Task 6: Ablation — does adding real BFS map control improve WP?

We restrict training to the subset of matches we have BFS coverage for,
and compare two models per map:

  * Baseline: [round_time_pct, ct_alive, t_alive, hp_ct, hp_t, equip_ct,
    equip_t, alive_diff, equip_diff, is_bomb_planted]
  * +MC:     baseline + [ct_map_control]

We only train a map if we have at least 5 matches with MC coverage.

Uses the SAME match-based GroupKFold split (same seed) for both models so
the comparison is paired. Reports per-map delta-AUC and delta-Brier.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

DB_PATH = Path("output/full_v2/cs2_analytics.duckdb")
OUT_DIR = Path("output/full_v2/analysis")
MODEL_DIR = OUT_DIR / "wp_models_ablation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MC_PATH = OUT_DIR / "map_control_ticks_sample.parquet"

BASE_FEATS = (
    "round_time_pct",
    "ct_alive",
    "t_alive",
    "hp_ct",
    "hp_t",
    "equip_ct",
    "equip_t",
    "alive_diff",
    "equip_diff",
    "is_bomb_planted",
)
MC_FEATS = BASE_FEATS + ("ct_map_control",)


def connect() -> duckdb.DuckDBPyConnection:
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


def build_features_for_matches(
    con: duckdb.DuckDBPyConnection, match_ids: list[str]
) -> pl.DataFrame:
    """Pull WP feature rows for the specified match_ids only, using direct
    file reads (faster than union_by_name glob)."""
    if not match_ids:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for mid in match_ids:
        tick_path = Path(f"output/full_v2/ticks/{mid}.parquet")
        round_path = Path(f"output/full_v2/rounds/{mid}.parquet")
        if not tick_path.exists() or not round_path.exists():
            continue
        sql = f"""
        WITH ticks_u AS (
            SELECT * FROM read_parquet('{tick_path.as_posix()}')
        ),
        rounds_u AS (
            SELECT * FROM read_parquet('{round_path.as_posix()}')
        ),
        agg AS (
            SELECT
                t.match_id, t.round_num, t.tick,
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
            a.*,
            r.winner,
            r.freeze_end,
            r."end" AS round_end,
            (a.tick - r.freeze_end) * 1.0 / NULLIF(r."end" - r.freeze_end, 0) AS round_time_pct
        FROM agg a
        JOIN rounds_u r ON r.match_id = a.match_id AND r.round_num = a.round_num
        WHERE a.tick >= r.freeze_end AND a.tick <= r."end"
          AND r.winner IS NOT NULL AND r."end" > r.freeze_end
        """
        frames.append(pl.from_arrow(con.execute(sql).arrow()))
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="vertical_relaxed")
    df = df.with_columns(
        [
            pl.col("round_time_pct").clip(0.0, 1.0),
            (pl.col("winner") == "ct").cast(pl.Int8).alias("ct_won"),
            (pl.col("ct_alive") - pl.col("t_alive")).alias("alive_diff"),
            (pl.col("equip_ct") - pl.col("equip_t")).alias("equip_diff"),
        ]
    )
    return df


def join_map_control(df: pl.DataFrame, mc: pl.DataFrame) -> pl.DataFrame:
    """Left-join BFS MC by (match_id, round_num, <nearest tick>).

    Strategy: sort both sides by tick, then use a nearest as-of join bounded
    to ±64 ticks (1 second). We also attach `has_mc` so downstream code can
    filter to rows with a real BFS score.
    """
    # join_asof requires BOTH sides to be globally sorted by the asof key.
    feats = df.sort(by="tick")
    mc_small = (
        mc.select(["match_id", "round_num", "tick", "ct_map_control"]).sort(
            by="tick"
        )
    )
    joined = feats.join_asof(
        mc_small,
        on="tick",
        by=["match_id", "round_num"],
        strategy="nearest",
        tolerance=64,
    )
    joined = joined.with_columns(
        [
            pl.col("ct_map_control").is_not_null().alias("has_mc"),
            pl.col("ct_map_control").fill_null(0.5),
        ]
    )
    return joined


def train_ablation(
    df: pl.DataFrame, map_name: str, require_mc: bool = False
) -> dict | None:
    import xgboost as xgb

    if require_mc:
        df = df.filter(pl.col("has_mc"))
        if len(df) < 300:
            return None

    groups = df["match_id"].to_numpy()
    if np.unique(groups).size < 5:
        return None
    y = df["ct_won"].to_numpy().astype(np.int32)
    X_base = df.select(BASE_FEATS).to_numpy().astype(np.float32)
    X_mc = df.select(MC_FEATS).to_numpy().astype(np.float32)

    gkf = GroupKFold(n_splits=min(5, np.unique(groups).size))
    base_aucs, mc_aucs = [], []
    base_briers, mc_briers = [], []

    xgb_args = dict(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=max((os.cpu_count() or 4) - 2, 2),
        random_state=42,
    )

    for fold, (tr, va) in enumerate(gkf.split(X_base, y, groups=groups)):
        m_base = xgb.XGBClassifier(**xgb_args)
        m_base.fit(X_base[tr], y[tr], verbose=False)
        p_base = m_base.predict_proba(X_base[va])[:, 1]
        base_aucs.append(float(roc_auc_score(y[va], p_base)))
        base_briers.append(float(brier_score_loss(y[va], p_base)))

        m_mc = xgb.XGBClassifier(**xgb_args)
        m_mc.fit(X_mc[tr], y[tr], verbose=False)
        p_mc = m_mc.predict_proba(X_mc[va])[:, 1]
        mc_aucs.append(float(roc_auc_score(y[va], p_mc)))
        mc_briers.append(float(brier_score_loss(y[va], p_mc)))
        print(
            f"  [{map_name} fold {fold}] "
            f"base_auc={base_aucs[-1]:.4f} +mc_auc={mc_aucs[-1]:.4f}  "
            f"d_auc={(mc_aucs[-1] - base_aucs[-1]):+.4f}"
        )

    return {
        "map_name": map_name,
        "n_rows": len(df),
        "n_matches": int(np.unique(groups).size),
        "mc_coverage": float((df["ct_map_control"] != 0.5).mean()),
        "base_auc": float(np.mean(base_aucs)),
        "mc_auc": float(np.mean(mc_aucs)),
        "delta_auc": float(np.mean(mc_aucs) - np.mean(base_aucs)),
        "base_brier": float(np.mean(base_briers)),
        "mc_brier": float(np.mean(mc_briers)),
        "delta_brier": float(np.mean(mc_briers) - np.mean(base_briers)),
        "feature_importance_mc": None,  # filled below
    }


def main() -> None:
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    con = connect()
    mc = pl.read_parquet(MC_PATH)
    parsed = pl.from_arrow(
        con.execute("SELECT match_id, map_name FROM parsed_demos").arrow()
    )
    mc = mc.join(parsed, on="match_id", how="inner")

    print(f"[ablation] mc rows={len(mc):,}  matches={mc['match_id'].n_unique()}")
    results = []
    for map_name in (
        mc.group_by("map_name")
        .agg(pl.col("match_id").n_unique().alias("n"))
        .sort("n", descending=True)["map_name"]
        .to_list()
    ):
        mc_match_ids = (
            mc.filter(pl.col("map_name") == map_name)["match_id"].unique().to_list()
        )
        print(
            f"\n[ablation] {map_name}: {len(mc_match_ids)} matches with BFS coverage"
        )
        if len(mc_match_ids) < 5:
            print(f"  skipping — too few matches")
            continue
        feats = build_features_for_matches(con, mc_match_ids)
        print(f"  fetched {len(feats):,} feature rows")
        if feats.is_empty():
            continue
        feats = feats.filter(pl.col("winner").is_not_null())
        enriched = join_map_control(feats, mc.filter(pl.col("map_name") == map_name))
        mc_frac = float(
            (enriched["ct_map_control"] != 0.5).cast(pl.Float64).mean()
        )
        print(f"  MC coverage: {mc_frac:.1%}")
        result = train_ablation(enriched, map_name, require_mc=False)
        if result is not None:
            result["mc_coverage"] = mc_frac
            result["mode"] = "all_rows"
            results.append(result)
            print(
                f"[{map_name}] all-rows base_auc={result['base_auc']:.4f} "
                f"mc_auc={result['mc_auc']:.4f} "
                f"d_auc={result['delta_auc']:+.4f} "
                f"d_brier={result['delta_brier']:+.4f}"
            )
        mc_only = train_ablation(enriched, map_name, require_mc=True)
        if mc_only is not None:
            mc_only["mc_coverage"] = 1.0
            mc_only["mode"] = "mc_only"
            results.append(mc_only)
            print(
                f"[{map_name}] mc-only  base_auc={mc_only['base_auc']:.4f} "
                f"mc_auc={mc_only['mc_auc']:.4f} "
                f"d_auc={mc_only['delta_auc']:+.4f} "
                f"d_brier={mc_only['delta_brier']:+.4f}"
            )

    (MODEL_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[ablation] wrote {MODEL_DIR / 'ablation_results.json'}")

    header = "| map | mode | n_rows | mc_cov | base_auc | +mc_auc | d_auc | base_brier | +mc_brier | d_brier |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in results:
        lines.append(
            f"| {r['map_name']} | {r.get('mode', '-')} | {r['n_rows']:,} | "
            f"{r['mc_coverage']:.1%} | "
            f"{r['base_auc']:.4f} | {r['mc_auc']:.4f} | {r['delta_auc']:+.4f} | "
            f"{r['base_brier']:.4f} | {r['mc_brier']:.4f} | {r['delta_brier']:+.4f} |"
        )
    summary = "\n".join(lines)
    print("\n## MC Ablation Summary\n" + summary)
    (OUT_DIR / "task6_mc_ablation.md").write_text(
        "# Task 6 — Does BFS map control improve WP?\n\n"
        f"Sample: {MC_PATH.name} ({len(mc):,} MC rows, 5 matches/map)\n\n"
        + summary
        + "\n"
    )


if __name__ == "__main__":
    main()
