"""Task 3: Map-specific Win Probability models.

Trains one XGBoost model per map using round-level tick aggregates with
match-based GroupKFold (5 folds). Reports per-map AUC and Brier loss.

Leaves a stub for the `ct_map_control` column: when real BFS features are
precomputed into `output/full_v2/analysis/map_control_ticks.parquet`, they
are joined in by (match_id, round_num, tick). If missing, the column is
filled with 0.0 and the model simply ignores it.

Usage:
    uv run python scripts/train_wp_model_v2.py

Writes:
    output/full_v2/analysis/wp_models/<map>.pkl (trained XGBoost)
    output/full_v2/analysis/wp_models/metrics.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

DB_PATH = Path("output/full_v2/cs2_analytics.duckdb")
OUT_DIR = Path("output/full_v2/analysis")
MODEL_DIR = OUT_DIR / "wp_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_DEMOS = 30  # minimum demos per map to train
DOWNSAMPLE_STRIDE = 20  # keep every Nth ordered tick row within a round (~5s grid)

FEATURE_COLS: tuple[str, ...] = (
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
    "ct_map_control",  # stubbed to 0.0 until BFS precomputed
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


TICK_FEATURES_SQL = """
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
),
ranked AS (
    SELECT
        a.*,
        ROW_NUMBER() OVER (
            PARTITION BY a.match_id, a.round_num ORDER BY a.tick
        ) AS rn,
        COUNT(*) OVER (PARTITION BY a.match_id, a.round_num) AS round_len
    FROM agg a
)
SELECT
    a.match_id,
    a.round_num,
    a.tick,
    pd.map_name,
    r.winner,
    r.freeze_end,
    r."end" AS round_end,
    a.ct_alive,
    a.t_alive,
    a.hp_ct,
    a.hp_t,
    a.equip_ct,
    a.equip_t,
    a.is_bomb_planted,
    (a.tick - r.freeze_end) * 1.0
        / NULLIF(r."end" - r.freeze_end, 0) AS round_time_pct
FROM ranked a
JOIN rounds_u r
  ON r.match_id = a.match_id AND r.round_num = a.round_num
JOIN parsed_demos pd
  ON pd.match_id = a.match_id
WHERE a.tick >= r.freeze_end
  AND a.tick <= r."end"
  AND r.winner IS NOT NULL
  AND r."end" > r.freeze_end
  /* Keep every {stride}'th sampled tick within the round, plus the very last one */
  AND ((a.rn - 1) % {stride} = 0 OR a.rn = a.round_len)
"""


def build_feature_frame(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    sql = TICK_FEATURES_SQL.format(stride=DOWNSAMPLE_STRIDE)
    arrow = con.execute(sql).arrow()
    df = pl.from_arrow(arrow)
    df = df.with_columns(
        [
            pl.col("round_time_pct").clip(0.0, 1.0),
            (pl.col("winner") == "ct").cast(pl.Int8).alias("ct_won"),
            (pl.col("ct_alive") - pl.col("t_alive")).alias("alive_diff"),
            (pl.col("equip_ct") - pl.col("equip_t")).alias("equip_diff"),
        ]
    )
    return df


def attach_map_control(df: pl.DataFrame) -> pl.DataFrame:
    """Join precomputed BFS map-control scores if available; else stub 0.0."""
    mc_path = OUT_DIR / "map_control_ticks.parquet"
    if mc_path.exists():
        mc = pl.read_parquet(mc_path)
        df = df.join(
            mc.select("match_id", "round_num", "tick", "ct_map_control"),
            on=["match_id", "round_num", "tick"],
            how="left",
        )
        df = df.with_columns(pl.col("ct_map_control").fill_null(0.0))
        print(
            f"[attach_map_control] joined {mc_path} ({len(mc):,} rows) — "
            f"non-null ticks: {df['ct_map_control'].is_not_null().sum()}"
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("ct_map_control"))
        print(f"[attach_map_control] stub ct_map_control=0.0 ({mc_path} not found)")
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


@dataclass
class MapResult:
    map_name: str
    n_rounds: int
    n_ticks: int
    n_matches: int
    fold_auc: list[float] = field(default_factory=list)
    fold_brier: list[float] = field(default_factory=list)
    fold_ll: list[float] = field(default_factory=list)
    feature_importance: dict[str, float] | None = None

    def mean_auc(self) -> float:
        return float(np.mean(self.fold_auc)) if self.fold_auc else float("nan")

    def mean_brier(self) -> float:
        return float(np.mean(self.fold_brier)) if self.fold_brier else float("nan")

    def to_dict(self) -> dict:
        return {
            "map_name": self.map_name,
            "n_rounds": self.n_rounds,
            "n_ticks": self.n_ticks,
            "n_matches": self.n_matches,
            "mean_auc": self.mean_auc(),
            "mean_brier": self.mean_brier(),
            "fold_auc": self.fold_auc,
            "fold_brier": self.fold_brier,
            "fold_log_loss": self.fold_ll,
            "feature_importance": self.feature_importance,
        }


def train_one_map(map_df: pl.DataFrame, map_name: str) -> MapResult | None:
    import xgboost as xgb  # local import so the file can parse without xgb

    match_ids = map_df["match_id"].unique()
    if len(match_ids) < MIN_DEMOS:
        print(
            f"[skip] {map_name} has only {len(match_ids)} matches (< {MIN_DEMOS})"
        )
        return None

    X = map_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = map_df["ct_won"].to_numpy().astype(np.int32)
    groups = map_df["match_id"].to_numpy()
    n_unique_matches = int(np.unique(groups).size)
    n_rounds_est = map_df.select(
        pl.concat_str([pl.col("match_id"), pl.col("round_num").cast(pl.Utf8)], separator=":")
    ).unique().height

    result = MapResult(
        map_name=map_name,
        n_rounds=n_rounds_est,
        n_ticks=len(map_df),
        n_matches=n_unique_matches,
    )

    gkf = GroupKFold(n_splits=min(5, n_unique_matches))
    importances = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        Xtr, ytr = X[train_idx], y[train_idx]
        Xva, yva = X[val_idx], y[val_idx]
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=max((os.cpu_count() or 4) - 2, 2),
            random_state=42,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        proba = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, proba)
        brier = brier_score_loss(yva, proba)
        # log loss, clip to avoid -inf
        p = np.clip(proba, 1e-6, 1 - 1e-6)
        ll = -float(np.mean(yva * np.log(p) + (1 - yva) * np.log(1 - p)))
        result.fold_auc.append(float(auc))
        result.fold_brier.append(float(brier))
        result.fold_ll.append(ll)
        importances += model.feature_importances_
        print(
            f"  [{map_name} fold {fold}] auc={auc:.4f} brier={brier:.4f} logloss={ll:.4f}"
        )

    # Retrain on all data for the final saved model.
    final = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=max((os.cpu_count() or 4) - 2, 2),
        random_state=42,
    )
    final.fit(X, y, verbose=False)
    model_path = MODEL_DIR / f"{map_name}.json"
    final.save_model(model_path)
    print(f"[{map_name}] saved {model_path}")

    importances /= max(gkf.get_n_splits(), 1)
    result.feature_importance = {
        name: float(val) for name, val in zip(FEATURE_COLS, importances)
    }
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    con = connect()
    print("[wp] pulling features...", flush=True)
    t0 = time.time()
    df = build_feature_frame(con)
    print(f"[wp] {len(df):,} rows, {len(df['match_id'].unique())} matches in {time.time() - t0:.1f}s")
    df = attach_map_control(df)

    all_results: list[MapResult] = []
    map_counts = (
        df.group_by("map_name")
        .agg(pl.col("match_id").n_unique().alias("n_matches"))
        .sort("n_matches", descending=True)
    )
    print("\n[wp] per-map demo counts:")
    print(map_counts)

    for map_name in map_counts["map_name"].to_list():
        map_df = df.filter(pl.col("map_name") == map_name)
        result = train_one_map(map_df, map_name)
        if result is not None:
            all_results.append(result)
            print(
                f"[{map_name}] mean AUC={result.mean_auc():.4f} "
                f"brier={result.mean_brier():.4f}"
            )

    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps([r.to_dict() for r in all_results], indent=2)
    )
    print(f"\n[wp] wrote {metrics_path}")

    print("\n## Per-map WP model summary")
    header = "| map | matches | rounds | ticks | mean_auc | mean_brier | top_feature |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in all_results:
        top = max(r.feature_importance.items(), key=lambda kv: kv[1]) if r.feature_importance else ("-", 0)
        lines.append(
            f"| {r.map_name} | {r.n_matches} | {r.n_rounds} | {r.n_ticks:,} | "
            f"{r.mean_auc():.4f} | {r.mean_brier():.4f} | {top[0]} ({top[1]:.2f}) |"
        )
    summary = "\n".join(lines)
    print(summary)
    (OUT_DIR / "task3_wp_summary.md").write_text(
        "# Task 3 — Per-map WP models\n\n" + summary + "\n"
    )


if __name__ == "__main__":
    main()
