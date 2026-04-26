"""Task 4: Precompute real BFS map-control scores at key moments.

Strategy: only compute BFS at *kill tick* and *round freeze_end* — that
gives us ~10-30 samples per round, ~160k-500k BFS calls for the full
dataset. Each BFS call is ~5ms so ~30-60 minutes.

Implementation notes:
  * Nav meshes are lazy-loaded via `awpy.stats.map_control._get_nav`.
  * The BFS cache (`_conical_bfs_cached`) is per-process and very effective
    if the same source tile is seen for many ticks — which is common with
    holding angles.
  * We write to a Parquet file: one row per (match_id, round_num, tick)
    with `ct_map_control` scalar + tile counts for diagnostics.

Usage examples:
    # Compute a tiny sample (1 demo per map) to verify the pipeline
    uv run python scripts/wp_task4_bfs_precompute.py --sample 1

    # Full precompute for all matches on kill ticks + freeze_end
    uv run python scripts/wp_task4_bfs_precompute.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from awpy.stats.map_control import (
    MapControlConfig,
    get_map_control_tick_metric,
)

DB_PATH = Path("output/full_v2/cs2_analytics.duckdb")
OUT_DIR = Path("output/full_v2/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT = OUT_DIR / "map_control_ticks.parquet"


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


TARGET_TICKS_SQL = """
WITH rounds_u AS (
    SELECT * FROM read_parquet(
        'output/full_v2/rounds/*.parquet', union_by_name = true
    )
),
kills_u AS (
    SELECT * FROM read_parquet(
        'output/full_v2/kills/*.parquet', union_by_name = true
    )
),
/* freeze_end + every kill tick per round */
kill_events AS (
    SELECT k.match_id, k.round_num, k.tick AS target_tick, 'kill' AS kind
    FROM kills_u k
    WHERE k.attacker_side IS NOT NULL
),
freeze_events AS (
    SELECT r.match_id, r.round_num, r.freeze_end AS target_tick, 'freeze' AS kind
    FROM rounds_u r
    WHERE r.freeze_end IS NOT NULL
),
end_events AS (
    SELECT r.match_id, r.round_num, r."end" AS target_tick, 'end' AS kind
    FROM rounds_u r
    WHERE r."end" IS NOT NULL
)
SELECT * FROM kill_events
UNION ALL SELECT * FROM freeze_events
UNION ALL SELECT * FROM end_events
"""


def pull_targets(
    con: duckdb.DuckDBPyConnection, map_filter: str | None, sample_per_map: int | None
) -> pl.DataFrame:
    """Return a DataFrame of (match_id, round_num, target_tick, kind, map_name).

    Only keeps demos that are also present in parsed_demos (joined on map_name).
    """
    events = pl.from_arrow(con.execute(TARGET_TICKS_SQL).arrow())
    # Join in map_name via parsed_demos
    parsed = pl.from_arrow(
        con.execute(
            "SELECT match_id, map_name FROM parsed_demos"
        ).arrow()
    )
    events = events.join(parsed, on="match_id", how="inner")
    if map_filter:
        events = events.filter(pl.col("map_name") == map_filter)

    if sample_per_map:
        # Take `sample_per_map` matches per map.
        kept_ids: list[str] = []
        for map_name, group in events.group_by("map_name"):
            mids = group["match_id"].unique().to_list()[:sample_per_map]
            kept_ids.extend(mids)
        events = events.filter(pl.col("match_id").is_in(kept_ids))
    return events.unique(subset=["match_id", "round_num", "target_tick"]).sort(
        ["map_name", "match_id", "round_num", "target_tick"]
    )


def nearest_sampled_tick(
    target_ticks: pl.DataFrame, sampled_ticks: np.ndarray
) -> np.ndarray:
    """For each target tick, find the nearest sampled tick within ±16 ticks.

    Returns an int array of the nearest sampled tick (or -1 if none in range).
    """
    # sampled_ticks is sorted ascending
    targets = target_ticks.to_numpy()
    idx = np.searchsorted(sampled_ticks, targets)
    res = np.full(targets.shape, -1, dtype=np.int64)
    for i, pos in enumerate(idx):
        candidates = []
        if pos > 0:
            candidates.append(sampled_ticks[pos - 1])
        if pos < len(sampled_ticks):
            candidates.append(sampled_ticks[pos])
        best = -1
        best_dist = 17
        for cand in candidates:
            d = abs(int(cand) - int(targets[i]))
            if d < best_dist:
                best = int(cand)
                best_dist = d
        res[i] = best
    return res


def fetch_match_ticks(
    con: duckdb.DuckDBPyConnection, match_id: str
) -> pl.DataFrame:
    """Load all sampled tick rows for a single match.

    Fast path: the ETL writes one parquet per match named by match_id.
    Read that file directly to avoid scanning 881 parquet files.
    """
    direct_path = Path(f"output/full_v2/ticks/{match_id}.parquet")
    if direct_path.exists():
        arrow = con.execute(
            f"""
            SELECT
                match_id, round_num, tick, steamid, name, side, X, Y, Z,
                health, armor, yaw, weapon_name, current_equip_value
            FROM read_parquet('{direct_path.as_posix()}')
            """
        ).arrow()
    else:
        arrow = con.execute(
            """
            SELECT
                match_id, round_num, tick, steamid, name, side, X, Y, Z,
                health, armor, yaw, weapon_name, current_equip_value
            FROM read_parquet('output/full_v2/ticks/*.parquet', union_by_name = true)
            WHERE match_id = ?
            """,
            [match_id],
        ).arrow()
    return pl.from_arrow(arrow)


def compute_for_match(
    con: duckdb.DuckDBPyConnection,
    match_id: str,
    map_name: str,
    targets: pl.DataFrame,
    config: MapControlConfig,
) -> pl.DataFrame:
    """Compute MC for all target ticks in a single match.

    Snaps each target tick to the nearest stored sampled tick (±16 ticks).
    """
    tdf = fetch_match_ticks(con, match_id)
    if tdf.is_empty():
        return pl.DataFrame()
    all_ticks = tdf["tick"].unique().sort().to_numpy()
    match_targets = targets.filter(pl.col("match_id") == match_id)
    snapped = nearest_sampled_tick(match_targets["target_tick"], all_ticks)
    match_targets = match_targets.with_columns(
        pl.Series("snapped_tick", snapped)
    ).filter(pl.col("snapped_tick") >= 0)

    # Rename columns for awpy/Polars API expected names.
    tdf = tdf.rename({"weapon_name": "weapon"})

    out_rows: list[dict] = []
    uniq_ticks = match_targets["snapped_tick"].unique().to_list()
    for snap_tick in uniq_ticks:
        frame = tdf.filter(pl.col("tick") == snap_tick)
        if frame.is_empty():
            continue
        try:
            ct_mc = get_map_control_tick_metric(map_name, frame, config=config)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] {match_id} tick {snap_tick}: {type(exc).__name__}: {exc}")
            ct_mc = float("nan")
        ct_alive = int(
            frame.filter((pl.col("side") == "ct") & (pl.col("health") > 0)).height
        )
        t_alive = int(
            frame.filter((pl.col("side") == "t") & (pl.col("health") > 0)).height
        )
        out_rows.append(
            {
                "match_id": match_id,
                "map_name": map_name,
                "snapped_tick": int(snap_tick),
                "ct_map_control": float(ct_mc),
                "ct_alive": ct_alive,
                "t_alive": t_alive,
            }
        )
    if not out_rows:
        return pl.DataFrame()
    computed = pl.DataFrame(out_rows)

    # Join back to target rows so we can record original kill/freeze/end context.
    joined = match_targets.join(
        computed.drop("match_id", "map_name"),
        on="snapped_tick",
        how="left",
    )
    return joined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="matches per map (0 = all)")
    ap.add_argument("--map", type=str, default=None, help="restrict to this map")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--conical", action="store_true", help="use conical BFS")
    args = ap.parse_args()

    con = connect()
    config = MapControlConfig(use_conical_bfs=args.conical)

    print("[bfs] pulling target ticks...")
    targets = pull_targets(
        con, map_filter=args.map, sample_per_map=args.sample or None
    )
    match_ids = targets["match_id"].unique().to_list()
    print(
        f"[bfs] {len(targets):,} target rows across {len(match_ids)} matches / "
        f"{targets['map_name'].n_unique()} maps"
    )

    chunks: list[pl.DataFrame] = []
    start = time.time()
    for i, mid in enumerate(match_ids):
        map_name = targets.filter(pl.col("match_id") == mid)["map_name"][0]
        t0 = time.time()
        chunk = compute_for_match(con, mid, map_name, targets, config)
        if not chunk.is_empty():
            chunks.append(chunk)
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (len(match_ids) - i - 1)
            print(
                f"[bfs] match {i + 1}/{len(match_ids)} {mid} ({map_name}) "
                f"{len(chunk)} rows in {time.time() - t0:.1f}s  "
                f"(elapsed={elapsed:.0f}s, eta={eta:.0f}s)"
            )

    if not chunks:
        print("[bfs] nothing computed")
        return

    full = pl.concat(chunks)
    # Write with the joined `tick` column renamed to tick for downstream use
    full = full.with_columns(
        pl.col("snapped_tick").alias("tick")
    )
    full.write_parquet(args.out)
    print(f"[bfs] wrote {args.out} ({len(full):,} rows)")


if __name__ == "__main__":
    main()
