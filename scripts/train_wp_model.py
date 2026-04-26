"""Win Probability Model — CS2 dust2 PoC.

Trains a gradient-boosted model on per-tick features derived from 100 demo matches.
Evaluates with match-based cross-validation to avoid leakage.

Phase 1 — Baseline + economy features:
  - n_alive_ct, n_alive_t
  - health_ct, health_t
  - equip_ct, equip_t   (total equipment value of alive players per side)
  - time_elapsed_pct  (0=round start, 1=round end)
  - bomb_planted       (1 if bomb is planted at this tick)

Phase 2 — + map_control (requires --with-map-control):
  - map_control  (0=T controls map, 1=CT controls map)
  Computed via vectorised BFS matrix: pre-compute all 2248 tile→tile BFS values once,
  then each tick is a few numpy sum operations.

Target:
  - ct_won (1 if CT won the round, else 0)

Usage:
    uv run python scripts/train_wp_model.py [--db path] [--output path] [--with-map-control]
"""

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import awpy.data
import awpy.nav
import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


BASELINE_FEATURE_COLS = [
    "n_alive_ct",
    "n_alive_t",
    "health_ct",
    "health_t",
    "equip_ct",
    "equip_t",
    "time_elapsed_pct",
    "bomb_planted",
]
TARGET_COL = "ct_won"

# ─── SQL Queries ──────────────────────────────────────────────────────────────

FEATURE_QUERY = """
WITH live_ticks AS (
    SELECT
        t.match_id,
        t.round_num,
        t.tick,
        t.side,
        t.health,
        COALESCE(t.current_equip_value, 0) AS current_equip_value,
        r.freeze_end,
        r.end          AS round_end,
        r.bomb_plant,
        r.winner
    FROM ticks t
    JOIN rounds r
        ON t.round_num = r.round_num
       AND t.match_id  = r.match_id
    WHERE t.tick >= r.freeze_end
      AND t.tick <= r.end
),
bin_repr AS (
    SELECT
        match_id,
        round_num,
        tick // 128              AS bin,
        MAX(tick)                AS repr_tick
    FROM live_ticks
    GROUP BY match_id, round_num, tick // 128
),
per_bin AS (
    SELECT
        lt.match_id,
        lt.round_num,
        br.bin,
        br.repr_tick,
        SUM(CASE WHEN lt.side = 'ct' AND lt.health > 0 THEN 1 ELSE 0 END)    AS n_alive_ct,
        SUM(CASE WHEN lt.side = 't'  AND lt.health > 0 THEN 1 ELSE 0 END)    AS n_alive_t,
        SUM(CASE WHEN lt.side = 'ct' THEN GREATEST(lt.health, 0) ELSE 0 END) AS health_ct,
        SUM(CASE WHEN lt.side = 't'  THEN GREATEST(lt.health, 0) ELSE 0 END) AS health_t,
        SUM(CASE WHEN lt.side = 'ct' AND lt.health > 0 THEN lt.current_equip_value ELSE 0 END) AS equip_ct,
        SUM(CASE WHEN lt.side = 't'  AND lt.health > 0 THEN lt.current_equip_value ELSE 0 END) AS equip_t,
        MIN(lt.freeze_end)  AS freeze_end,
        MIN(lt.round_end)   AS round_end,
        MIN(lt.bomb_plant)  AS bomb_plant,
        MIN(lt.winner)      AS winner
    FROM live_ticks lt
    JOIN bin_repr br
        ON lt.match_id  = br.match_id
       AND lt.round_num = br.round_num
       AND lt.tick      = br.repr_tick
    GROUP BY lt.match_id, lt.round_num, br.bin, br.repr_tick
)
SELECT
    match_id,
    round_num,
    bin,
    repr_tick,
    n_alive_ct,
    n_alive_t,
    health_ct,
    health_t,
    equip_ct,
    equip_t,
    CAST(repr_tick - freeze_end AS FLOAT)
        / GREATEST(CAST(round_end - freeze_end AS FLOAT), 1.0) AS time_elapsed_pct,
    CASE WHEN bomb_plant IS NOT NULL
              AND repr_tick >= bomb_plant THEN 1 ELSE 0 END     AS bomb_planted,
    CASE WHEN winner = 'ct' THEN 1 ELSE 0 END                  AS ct_won
FROM per_bin
ORDER BY match_id, round_num, bin
"""

# Separate query: player (X,Y,Z,side) at each repr_tick for map control
POSITIONS_QUERY = """
WITH live_ticks AS (
    SELECT
        t.match_id, t.round_num, t.tick, t.side, t.X, t.Y, t.Z,
        r.freeze_end, r.end AS round_end
    FROM ticks t
    JOIN rounds r
        ON t.round_num = r.round_num AND t.match_id = r.match_id
    WHERE t.tick >= r.freeze_end AND t.tick <= r.end
),
bin_repr AS (
    SELECT match_id, round_num, tick // 128 AS bin, MAX(tick) AS repr_tick
    FROM live_ticks
    GROUP BY match_id, round_num, tick // 128
)
SELECT lt.match_id, lt.round_num, br.bin, br.repr_tick,
       lt.side, lt.X, lt.Y, lt.Z
FROM live_ticks lt
JOIN bin_repr br
    ON lt.match_id  = br.match_id
   AND lt.round_num = br.round_num
   AND lt.tick      = br.repr_tick
ORDER BY lt.match_id, lt.round_num, br.bin
"""


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_features(db_path: str) -> pd.DataFrame:
    log.info("Loading features from %s", db_path)
    t0 = time.perf_counter()
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(FEATURE_QUERY).df()
    con.close()
    log.info("Loaded %d rows in %.1fs", len(df), time.perf_counter() - t0)
    return df


def load_positions(db_path: str) -> pd.DataFrame:
    log.info("Loading player positions for map control...")
    t0 = time.perf_counter()
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(POSITIONS_QUERY).df()
    con.close()
    log.info("Loaded %d player-tick rows in %.1fs", len(df), time.perf_counter() - t0)
    return df


# ─── Map Control Feature Engineering ─────────────────────────────────────────

def _bfs_from_tile(source_id: int, area_info: dict) -> dict[int, float]:
    """BFS from source_id → {tile_id: value}. Depth-limited to 10."""
    seen: set[int] = set()
    result: dict[int, float] = {}
    queue: deque = deque([(source_id, 1.0, 0)])
    while queue:
        tile_id, val, depth = queue.popleft()
        if tile_id in seen:
            continue
        seen.add(tile_id)
        result[tile_id] = val
        if depth < 9:
            next_val = (10 - (depth + 1)) / 10
            for nb in area_info[tile_id].connections:
                if nb not in seen and nb in area_info:
                    queue.append((nb, next_val, depth + 1))
    return result


def _polygon_area(corners) -> float:
    """Shoelace formula on (x, y) of corners."""
    pts = [(c.x, c.y) for c in corners]
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def precompute_mc_matrix(map_name: str = "de_dust2"):
    """
    Pre-compute BFS value matrix and tile areas for map_name.

    Returns
    -------
    tile_ids : list[int]
        Ordered list of area_ids (length N).
    tile_idx : dict[int, int]
        area_id → matrix row/col index.
    bfs_matrix : np.ndarray shape (N, N)
        bfs_matrix[src_idx, tgt_idx] = BFS value from src to tgt (0 if unreachable in 10 steps).
    tile_areas : np.ndarray shape (N,)
        Polygon area in map units for each tile.
    """
    log.info("Pre-computing BFS matrix for %s...", map_name)
    t0 = time.perf_counter()
    nav = awpy.nav.Nav.from_json(awpy.data.NAVS_DIR / f"{map_name}.json")
    area_info = nav.areas

    tile_ids = list(area_info.keys())
    N = len(tile_ids)
    tile_idx = {aid: i for i, aid in enumerate(tile_ids)}

    bfs_matrix = np.zeros((N, N), dtype=np.float32)
    for aid in tile_ids:
        bfs_result = _bfs_from_tile(aid, area_info)
        src_i = tile_idx[aid]
        for tgt_id, val in bfs_result.items():
            if tgt_id in tile_idx:
                bfs_matrix[src_i, tile_idx[tgt_id]] = val

    tile_areas = np.array(
        [_polygon_area(area_info[aid].corners) for aid in tile_ids],
        dtype=np.float32,
    )

    elapsed = time.perf_counter() - t0
    sparsity = (bfs_matrix == 0).mean()
    log.info(
        "BFS matrix: %d×%d tiles, sparsity=%.1f%%, elapsed=%.1fs",
        N, N, sparsity * 100, elapsed,
    )
    return tile_ids, tile_idx, bfs_matrix, tile_areas


def build_position_tree(nav):
    """Build BallTree over all nav corners for fast (X,Y,Z)→area_id lookup."""
    corners_xyz = []
    corner_aids = []
    for area in nav.areas.values():
        for corner in area.corners:
            corners_xyz.append([corner.x, corner.y, corner.z])
            corner_aids.append(area.area_id)
    corners_array = np.array(corners_xyz, dtype=np.float64)
    aids_array = np.array(corner_aids, dtype=np.int64)
    tree = BallTree(corners_array, metric="euclidean")
    return tree, aids_array


def batch_lookup_tiles(
    tree: BallTree,
    aids_array: np.ndarray,
    tile_idx: dict[int, int],
    xyz: np.ndarray,
) -> np.ndarray:
    """(N, 3) positions → (N,) tile matrix indices. Unknown tiles → -1."""
    _, indices = tree.query(xyz.astype(np.float64), k=1)
    area_ids = aids_array[indices.flatten()]
    return np.array([tile_idx.get(int(aid), -1) for aid in area_ids], dtype=np.int32)


def compute_map_control_features(
    pos_df: pd.DataFrame,
    tile_idx: dict[int, int],
    bfs_matrix: np.ndarray,
    tile_areas: np.ndarray,
    nav,
) -> pd.DataFrame:
    """
    Compute map_control metric for each (match_id, round_num, bin) tick.

    Returns DataFrame with columns [match_id, round_num, bin, map_control].
    """
    log.info("Building BallTree for position lookup...")
    t0 = time.perf_counter()
    tree, aids_array = build_position_tree(nav)

    # Batch tile lookup for all player positions
    xyz = pos_df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    log.info("Batch-looking up %d player positions...", len(xyz))
    pos_df = pos_df.copy()
    pos_df["tile_idx"] = batch_lookup_tiles(tree, aids_array, tile_idx, xyz)
    log.info("Position lookup done in %.1fs", time.perf_counter() - t0)

    N = len(bfs_matrix)
    log.info("Computing MC metric for %d tick-bins...", pos_df.groupby(["match_id", "round_num", "bin"]).ngroups)
    t0 = time.perf_counter()

    records = []
    for (match_id, round_num, bin_val), grp in pos_df.groupby(
        ["match_id", "round_num", "bin"], sort=False
    ):
        ct_idx = grp.loc[grp["side"] == "ct", "tile_idx"].values
        t_idx  = grp.loc[grp["side"] == "t",  "tile_idx"].values

        # Filter out invalid tile indices
        ct_idx = ct_idx[ct_idx >= 0]
        t_idx  = t_idx[t_idx  >= 0]

        if len(ct_idx) == 0 and len(t_idx) == 0:
            mc = 0.5
        elif len(ct_idx) == 0:
            mc = 0.0
        elif len(t_idx) == 0:
            mc = 1.0
        else:
            ct_vals = bfs_matrix[ct_idx].sum(axis=0)  # (N,)
            t_vals  = bfs_matrix[t_idx].sum(axis=0)   # (N,)
            total   = ct_vals + t_vals
            active  = total > 0
            if not active.any():
                mc = 0.5
            else:
                safe_total = np.where(active, total, 1.0)
                tile_metric = np.where(active, ct_vals / safe_total, 0.0)
                w = tile_areas * active
                mc = float((tile_metric * w).sum() / w.sum())

        records.append({"match_id": match_id, "round_num": round_num, "bin": bin_val, "map_control": mc})

    elapsed = time.perf_counter() - t0
    mc_df = pd.DataFrame.from_records(records)
    log.info("MC computation done: %d rows in %.1fs (%.0f ticks/s)", len(mc_df), elapsed, len(mc_df) / elapsed)
    return mc_df


# ─── Model Training ───────────────────────────────────────────────────────────

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 0.1,
    "tree_method": "hist",
    "seed": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def train_and_eval(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5) -> dict:
    """Group-k-fold CV by match_id. Returns metrics dict."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.int32)
    groups = pd.Categorical(df["match_id"]).codes

    gkf = GroupKFold(n_splits=n_splits)
    aucs, lls = [], []
    oof_pred = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        oof_pred[val_idx] = preds
        auc = roc_auc_score(y[val_idx], preds)
        ll  = log_loss(y[val_idx], preds)
        aucs.append(auc)
        lls.append(ll)
        log.info(
            "  fold %d | AUC=%.4f | LogLoss=%.4f | val_matches=%d",
            fold + 1, auc, ll, len(np.unique(groups[val_idx])),
        )

    oof_auc = roc_auc_score(y, oof_pred)
    oof_ll  = log_loss(y, oof_pred)
    log.info("OOF AUC=%.4f ± %.4f | LogLoss=%.4f", np.mean(aucs), np.std(aucs), oof_ll)

    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y)

    return {
        "fold_aucs": aucs,
        "fold_lls": lls,
        "oof_auc": oof_auc,
        "oof_ll": oof_ll,
        "final_model": final_model,
        "oof_pred": oof_pred,
        "y": y,
        "feature_cols": feature_cols,
    }


def print_feature_importance(model: xgb.XGBClassifier, feature_cols: list[str]) -> None:
    scores = model.feature_importances_
    log.info("Feature importances:")
    for name, score in sorted(zip(feature_cols, scores), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        log.info("  %-20s %.4f  %s", name, score, bar)


def print_calibration(df: pd.DataFrame, oof_pred: np.ndarray, y: np.ndarray, label: str = "") -> None:
    dfc = df.copy()
    dfc["pred"] = oof_pred
    dfc["y"] = y
    dfc["time_bucket"] = pd.cut(
        dfc["time_elapsed_pct"],
        bins=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
    )
    prefix = f"[{label}] " if label else ""
    log.info("%sAUC by time bucket:", prefix)
    for bucket, grp in dfc.groupby("time_bucket", observed=True):
        if len(grp) < 100:
            continue
        try:
            auc = roc_auc_score(grp["y"], grp["pred"])
        except Exception:
            auc = float("nan")
        log.info("  t=%s  n=%6d  AUC=%.4f", bucket, len(grp), auc)


def summarize(label: str, results: dict, df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("%s RESULTS", label)
    log.info("  Features:  %s", ", ".join(results["feature_cols"]))
    log.info("  Matches:   %d", df["match_id"].nunique())
    log.info("  Rounds:    %d", df.groupby(["match_id", "round_num"]).ngroups)
    log.info("  Samples:   %d", len(df))
    log.info("  OOF AUC:   %.4f ± %.4f", np.mean(results["fold_aucs"]), np.std(results["fold_aucs"]))
    log.info("  OOF LogL:  %.4f", results["oof_ll"])
    log.info("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train WP model (baseline + optional map control)")
    parser.add_argument("--db", default="output/full_dust2/cs2_analytics.duckdb")
    parser.add_argument("--output", default="output/full_dust2/wp_model")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--with-map-control", action="store_true",
                        help="Add map_control feature (Phase 2)")
    parser.add_argument("--map-name", default="de_dust2",
                        help="Map to pre-compute BFS for (default: de_dust2)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Baseline ──────────────────────────────────────────────────
    df = load_features(args.db)

    assert df[TARGET_COL].isin([0, 1]).all()
    log.info(
        "Dataset: %d samples | %d matches | %d rounds | ct_won=%.1f%%",
        len(df), df["match_id"].nunique(),
        df.groupby(["match_id", "round_num"]).ngroups,
        df[TARGET_COL].mean() * 100,
    )

    log.info("── Phase 1: Baseline ──────────────────────────────")
    log.info("Training with %d-fold GroupKFold (groups=match_id)...", args.folds)
    baseline_results = train_and_eval(df, BASELINE_FEATURE_COLS, n_splits=args.folds)
    print_feature_importance(baseline_results["final_model"], BASELINE_FEATURE_COLS)
    print_calibration(df, baseline_results["oof_pred"], baseline_results["y"], label="baseline")
    summarize("BASELINE", baseline_results, df)

    # Save baseline OOF
    oof_df = df[["match_id", "round_num", "bin", "repr_tick", "time_elapsed_pct"]].copy()
    oof_df["pred_baseline"] = baseline_results["oof_pred"]
    oof_df["ct_won"] = baseline_results["y"]

    if not args.with_map_control:
        oof_df.to_parquet(str(output_dir / "oof_predictions.parquet"), index=False)
        baseline_results["final_model"].save_model(str(output_dir / "wp_baseline.ubj"))
        log.info("Saved baseline model and OOF predictions.")
        return

    # ── Phase 2: + Map Control ─────────────────────────────────────────────
    log.info("── Phase 2: + Map Control ─────────────────────────")

    # Pre-compute BFS matrix
    tile_ids, tile_idx, bfs_matrix, tile_areas = precompute_mc_matrix(args.map_name)

    # Load player positions
    pos_df = load_positions(args.db)

    # Load nav for BallTree
    nav = awpy.nav.Nav.from_json(awpy.data.NAVS_DIR / f"{args.map_name}.json")

    # Compute MC features
    mc_df = compute_map_control_features(pos_df, tile_idx, bfs_matrix, tile_areas, nav)

    # Join to feature df
    df2 = df.merge(mc_df, on=["match_id", "round_num", "bin"], how="left")
    n_missing = df2["map_control"].isna().sum()
    if n_missing > 0:
        log.warning("%d rows missing map_control — filling with 0.5", n_missing)
        df2["map_control"] = df2["map_control"].fillna(0.5)

    mc_feature_cols = BASELINE_FEATURE_COLS + ["map_control"]
    log.info("Training Phase 2 model with %d features...", len(mc_feature_cols))
    mc_results = train_and_eval(df2, mc_feature_cols, n_splits=args.folds)
    print_feature_importance(mc_results["final_model"], mc_feature_cols)
    print_calibration(df2, mc_results["oof_pred"], mc_results["y"], label="+map_control")
    summarize("+MAP_CONTROL", mc_results, df2)

    # Delta AUC
    delta = mc_results["oof_auc"] - baseline_results["oof_auc"]
    log.info("Map control AUC lift: %+.4f (%.4f → %.4f)", delta,
             baseline_results["oof_auc"], mc_results["oof_auc"])

    # Save
    oof_df["pred_mc"] = mc_results["oof_pred"]
    oof_df.to_parquet(str(output_dir / "oof_predictions.parquet"), index=False)
    mc_results["final_model"].save_model(str(output_dir / "wp_with_mc.ubj"))
    baseline_results["final_model"].save_model(str(output_dir / "wp_baseline.ubj"))
    log.info("Saved all model artefacts to %s", output_dir)


if __name__ == "__main__":
    main()
