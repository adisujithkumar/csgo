#!/usr/bin/env python3
"""ETL Pipeline: Parse CS2 demos -> Parquet -> DuckDB.

Parses CS2 .dem files using awpy, transforms the data (sample ticks, add
match_id, normalize columns), saves as Parquet files, and registers them in a
DuckDB database for analytics queries.

Performance features:
    * Per-match parquet files (eliminates O(n^2) rewrite on every demo)
    * Multiprocessing worker pool (--workers N)
    * Skip flags for expensive data (--skip-grenades, --skip-damages,
      --skip-shots, --minimal)
    * Map filter (--map-filter de_dust2,de_mirage)
    * Max demos cap (--max-demos N)
    * Map discovery log (output/map_discovery.txt)

Usage:
    # Process specific .dem files with 4 workers
    python scripts/etl_pipeline.py --input /path/to/demos/*.dem \\
        --output /path/to/output --workers 4

    # Process .rar archives, only de_dust2, minimal mode, capped at 10 demos
    python scripts/etl_pipeline.py --input /path/to/archives/ \\
        --output /path/to/output --workers 4 --minimal \\
        --map-filter de_dust2 --max-demos 10

    # Validate existing output
    python scripts/etl_pipeline.py --validate --output /path/to/output

    # Process with custom tick sampling rate
    python scripts/etl_pipeline.py --input /path/to/*.dem \\
        --output /path/to/output --tick-rate 32
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_SAMPLE_INTERVAL = 16  # Sample every 16th tick (4 per second at 64-tick)
KNOWN_BAD_DEMOS = frozenset(
    {
        "imperial-vs-legacy-m3-nuke-p1.dem",
        "aurora-vs-heroic-m1-train-p2.dem",
    }
)

# All competitive CS2 maps.  Used for cheap filename-based map detection.
KNOWN_MAPS: tuple[str, ...] = (
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
    "de_vertigo",
    "de_cache",
    "de_cobblestone",
)

# Columns we always want from demo.ticks after parse()
TICK_KEEP_COLUMNS = [
    "tick",
    "steamid",
    "name",
    "round_num",
    "side",
    "X",
    "Y",
    "Z",
    "health",
    "place",
]

# Per-match parquet tables (one file per match_id per table).
PER_MATCH_TABLES: tuple[str, ...] = (
    "rounds",
    "kills",
    "damages",
    "bomb",
    "smokes",
    "infernos",
    "grenades",
    "shots",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """User-facing flags controlling what gets parsed and stored."""

    tick_interval: int = TICK_SAMPLE_INTERVAL
    skip_grenades: bool = False
    skip_damages: bool = False
    skip_shots: bool = False
    map_filter: frozenset[str] = field(default_factory=frozenset)

    @property
    def min_events(self) -> list[str]:
        """Return the minimum event list given the skip flags.

        The goal is to avoid parsing expensive per-tick events when they
        aren't needed.  Some events are mandatory because awpy uses them
        internally to build the rounds table:
          - round_freeze_end, round_officially_ended
          - player_spawn, player_given_c4, bomb_pickup
          - bomb_dropped, bomb_planted, bomb_defused, bomb_exploded
          - player_death (required for kills table)
          - smokegrenade_{detonate,expired}, inferno_{startburn,expire}
        """
        events = [
            "round_freeze_end",
            "round_officially_ended",
            "player_spawn",
            "player_given_c4",
            "bomb_pickup",
            "item_pickup",
            "player_death",
            "bomb_dropped",
            "bomb_planted",
            "bomb_defused",
            "bomb_exploded",
            "smokegrenade_detonate",
            "smokegrenade_expired",
            "inferno_startburn",
            "inferno_expire",
            # Always include flashbang/hegrenade detonate as they're cheap.
            "flashbang_detonate",
            "hegrenade_detonate",
        ]
        if not self.skip_damages:
            events.append("player_hurt")
        if not self.skip_shots:
            events.append("weapon_fire")
        return events


@dataclass
class DemoResult:
    """Result of processing a single demo file."""

    path: str
    match_id: str
    map_name: str = ""
    success: bool = False
    error: str = ""
    n_rounds: int = 0
    n_ticks_raw: int = 0
    n_ticks_sampled: int = 0
    n_kills: int = 0
    n_damages: int = 0
    n_bomb_events: int = 0
    elapsed_secs: float = 0.0
    # Row for matches.parquet.  Workers produce this so the parent can
    # batch-write a single shared matches.parquet at the end.
    match_row: dict | None = None
    # The source file the parent should log to map_discovery.txt.  For
    # .rar-extracted demos this is the archive filename.
    source_file: str = ""
    # "ok", "skipped", "failed", "filtered".
    status: str = "ok"


@dataclass
class PipelineReport:
    """Aggregate report for the full ETL run."""

    results: list[DemoResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def succeeded(self) -> list[DemoResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> list[DemoResult]:
        return [r for r in self.results if not r.success and r.status == "failed"]

    @property
    def filtered(self) -> list[DemoResult]:
        return [r for r in self.results if r.status == "filtered"]

    def print_summary(self) -> None:
        elapsed = self.end_time - self.start_time
        log.info("=" * 70)
        log.info("ETL Pipeline Complete")
        log.info("=" * 70)
        log.info(
            "Processed %d demo(s) in %.1fs  |  %d ok  |  %d failed  |  %d filtered",
            len(self.results),
            elapsed,
            len(self.succeeded),
            len(self.failed),
            len(self.filtered),
        )
        if self.succeeded:
            total_ticks = sum(r.n_ticks_sampled for r in self.succeeded)
            total_kills = sum(r.n_kills for r in self.succeeded)
            total_rounds = sum(r.n_rounds for r in self.succeeded)
            log.info(
                "Totals: %d rounds, %d sampled ticks, %d kills",
                total_rounds,
                total_ticks,
                total_kills,
            )
        if self.failed:
            log.warning("Failed demos:")
            for r in self.failed:
                log.warning("  %s -- %s", Path(r.path).name, r.error)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def match_id_from_path(path: Path) -> str:
    """Derive a deterministic match_id from the demo filename.

    Uses the stem (filename without extension) to produce a stable 12-char hex
    hash so that re-running the pipeline on the same file yields the same id.
    """
    return hashlib.sha256(path.stem.encode()).hexdigest()[:12]


def detect_map_from_filename(filename: str) -> str | None:
    """Detect a CS2 map name from a file's *filename* (case insensitive).

    Looks for any of ``KNOWN_MAPS`` (or the short form without the ``de_``
    prefix) in the filename.  Returns the canonical ``de_<name>`` string or
    None if no map can be detected.

    Examples:
        ``spirit-vs-ninjas-in-pyjamas-m2-dust2.dem`` -> ``de_dust2``
        ``aurora-vs-heroic-m3-mirage.dem``           -> ``de_mirage``
        ``blasttv-austin-major-2025-..-dust2.rar``    -> ``de_dust2``
    """
    name_lower = filename.lower()
    for full in KNOWN_MAPS:
        short = full.removeprefix("de_")
        # Word-ish boundary: allow hyphens, dots, underscores around the map.
        pattern = rf"(?<![a-z]){re.escape(short)}(?![a-z])"
        if re.search(pattern, name_lower):
            return full
    return None


def _available_columns(df: pl.DataFrame, wanted: list[str]) -> list[str]:
    """Return the subset of *wanted* that actually exist in *df*."""
    existing = set(df.columns)
    return [c for c in wanted if c in existing]


def extract_rar(rar_path: Path, dest_dir: Path) -> list[Path]:
    """Extract .dem files from a .rar archive into *dest_dir*.

    Returns a list of extracted .dem file paths.  Raises RuntimeError if
    unrar is not installed or extraction fails.
    """
    try:
        subprocess.run(
            ["unrar", "e", "-o+", str(rar_path), "*.dem", str(dest_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        msg = "unrar is not installed -- cannot extract .rar archives"
        raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        msg = f"unrar failed for {rar_path.name}: {exc.stderr.strip()}"
        raise RuntimeError(msg) from exc

    return sorted(dest_dir.glob("*.dem"))


def list_rar_dem_files(rar_path: Path) -> list[str]:
    """List .dem filenames inside a .rar archive without extracting.

    Uses ``unrar lb`` (bare listing) which is cheap (~100ms).  Returns a
    list of relative filenames.  Returns an empty list on failure so that
    the caller can fall back to extracting + parsing.
    """
    try:
        result = subprocess.run(
            ["unrar", "lb", str(rar_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().lower().endswith(".dem")
    ]


def resolve_input_files(patterns: list[str]) -> list[Path]:
    """Expand shell globs / literal paths into a deduplicated list of files.

    Accepts both .dem and .rar paths.  If a path is a directory, recurse
    one level and pick up any *.dem / *.rar inside.  Duplicate paths are
    removed while preserving order.
    """
    seen: set[Path] = set()
    result: list[Path] = []

    def _add(candidate: Path) -> None:
        try:
            canon = candidate.resolve()
        except OSError:
            return
        if canon not in seen:
            seen.add(canon)
            result.append(candidate)

    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            # Directory: pick up .dem and .rar files (non-recursive).
            for ext in ("*.dem", "*.rar"):
                for f in sorted(p.glob(ext)):
                    _add(f)
            continue

        candidates = [p] if p.exists() else sorted(p.parent.glob(p.name))
        for c in candidates:
            _add(c)
    return result


# ---------------------------------------------------------------------------
# Extract  (parse demo -> Polars DataFrames)
# ---------------------------------------------------------------------------


def parse_demo(
    dem_path: Path,
    config: PipelineConfig,
) -> dict[str, pl.DataFrame | dict]:
    """Parse a single .dem file and return raw Polars DataFrames.

    Returns a dict with keys: header, rounds, ticks, kills, damages, bomb,
    smokes, infernos, grenades, shots.  Skipped tables are empty DataFrames.

    Raises on parse failure so the caller can handle it.
    """
    from awpy import Demo  # noqa: PLC0415

    demo = Demo(path=dem_path)

    # Monkey-patch parse_grenades to be a no-op when grenades are skipped.
    # This avoids a significant chunk of parse time on large demos (~20-40%).
    if config.skip_grenades:
        def _empty_grenades() -> pl.DataFrame:
            return pl.DataFrame(
                schema={
                    "thrower_steamid": pl.Int64,
                    "thrower": pl.Utf8,
                    "grenade_type": pl.Utf8,
                    "tick": pl.Int32,
                    "X": pl.Float32,
                    "Y": pl.Float32,
                    "Z": pl.Float32,
                    "entity_id": pl.Int32,
                }
            )

        demo.parse_grenades = _empty_grenades  # type: ignore[method-assign]

    demo.parse(events=config.min_events)

    tables: dict[str, pl.DataFrame | dict] = {"header": demo.header}
    tables["rounds"] = demo.rounds
    tables["ticks"] = demo.ticks

    # Each cached-property may raise if the underlying events are absent
    # (or if we deliberately skipped the event).  We try each independently.
    accessors: list[tuple[str, object]] = [
        ("kills", lambda: demo.kills),
        ("bomb", lambda: demo.bomb),
        ("smokes", lambda: demo.smokes),
        ("infernos", lambda: demo.infernos),
    ]
    if not config.skip_damages:
        accessors.append(("damages", lambda: demo.damages))
    if not config.skip_shots:
        accessors.append(("shots", lambda: demo.shots))
    if not config.skip_grenades:
        accessors.append(("grenades", lambda: demo.grenades))

    for name, accessor in accessors:
        try:
            tables[name] = accessor()  # type: ignore[operator]
        except Exception:
            log.debug("Could not extract '%s' from %s", name, dem_path.name)
            tables[name] = pl.DataFrame()

    # Ensure all keys exist (empty frames for skipped tables).
    for key in ("kills", "damages", "bomb", "smokes", "infernos", "grenades", "shots"):
        tables.setdefault(key, pl.DataFrame())

    return tables


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def add_match_id(df: pl.DataFrame, match_id: str) -> pl.DataFrame:
    """Prepend a match_id column to a DataFrame."""
    if df.is_empty():
        return df
    return df.with_columns(pl.lit(match_id).alias("match_id"))


def sample_ticks(
    ticks: pl.DataFrame,
    interval: int = TICK_SAMPLE_INTERVAL,
) -> pl.DataFrame:
    """Downsample tick data: keep every *interval*-th tick, alive players only.

    Within each round we want deterministic, evenly-spaced snapshots.  We
    select the distinct tick values, sort them, keep every Nth one, then filter
    the full DataFrame to only rows whose tick is in that set AND the player
    is alive (health > 0).
    """
    if ticks.is_empty():
        return ticks

    # Deterministic sampling: rank ticks within each round and keep every Nth.
    tick_index = (
        ticks.select(["round_num", "tick"])
        .unique()
        .sort(["round_num", "tick"])
        .with_columns(
            pl.col("tick")
            .rank("ordinal")
            .over("round_num")
            .alias("_tick_rank")
        )
    )

    sampled_ticks = tick_index.filter(
        (pl.col("_tick_rank") - 1) % interval == 0
    ).select(["round_num", "tick"])

    # Semi-join: keep only rows whose (round_num, tick) pair is in sampled set.
    result = ticks.join(sampled_ticks, on=["round_num", "tick"], how="semi")

    # Alive players only.
    if "health" in result.columns:
        result = result.filter(pl.col("health") > 0)

    return result


def normalize_ticks(ticks: pl.DataFrame) -> pl.DataFrame:
    """Select and rename tick columns to a stable schema.

    If certain columns are absent in a particular demo (e.g. 'place'), the
    column is created with null values rather than crashing.
    """
    if ticks.is_empty():
        return ticks

    # Build the output column-by-column for resilience.
    exprs: list[pl.Expr] = []
    for col in TICK_KEEP_COLUMNS:
        if col in ticks.columns:
            exprs.append(pl.col(col))
        else:
            # Add a null column with the right name.
            exprs.append(pl.lit(None).alias(col))

    # Keep match_id if it was already added.
    if "match_id" in ticks.columns:
        exprs.insert(0, pl.col("match_id"))

    return ticks.select(exprs)


def transform_demo(
    raw: dict[str, pl.DataFrame | dict],
    match_id: str,
    tick_interval: int = TICK_SAMPLE_INTERVAL,
) -> dict[str, pl.DataFrame | dict]:
    """Apply all transformations to the raw parsed data.

    Returns a new dict with the same keys, DataFrames enriched with match_id,
    and ticks sampled + normalized.
    """
    out: dict[str, pl.DataFrame | dict] = {}

    # Header (dict, not a DataFrame) -- enrich with match_id.
    header = dict(raw["header"])  # shallow copy
    header["match_id"] = match_id
    out["header"] = header

    # Rounds.
    out["rounds"] = add_match_id(raw["rounds"], match_id)

    # Ticks: sample then normalize.
    ticks_sampled = sample_ticks(raw["ticks"], interval=tick_interval)
    ticks_normalized = normalize_ticks(ticks_sampled)
    out["ticks"] = add_match_id(ticks_normalized, match_id)

    # Other event tables: just add match_id.
    for table_name in ("kills", "damages", "bomb", "smokes", "infernos", "grenades", "shots"):
        df = raw.get(table_name, pl.DataFrame())
        if isinstance(df, pl.DataFrame):
            out[table_name] = add_match_id(df, match_id)
        else:
            out[table_name] = pl.DataFrame()

    return out


# ---------------------------------------------------------------------------
# Load  (write per-match Parquet files)
# ---------------------------------------------------------------------------


def write_parquet(
    transformed: dict[str, pl.DataFrame | dict],
    output_dir: Path,
    match_id: str,
    config: PipelineConfig,
) -> dict:
    """Write transformed DataFrames as per-match Parquet files.

    Layout::

        output_dir/
            matches.parquet            (shared, written by parent)
            ticks/{match_id}.parquet
            rounds/{match_id}.parquet
            kills/{match_id}.parquet
            damages/{match_id}.parquet
            bomb/{match_id}.parquet
            smokes/{match_id}.parquet
            infernos/{match_id}.parquet
            grenades/{match_id}.parquet
            shots/{match_id}.parquet

    Returns the match_row dict (to be appended to matches.parquet by the
    parent process).  Worker processes DO NOT touch matches.parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- matches row (returned, not written) ---
    header = transformed["header"]
    match_row = {
        "match_id": header.get("match_id", match_id),
        "map_name": header.get("map_name", ""),
        "server_name": header.get("server_name", ""),
        "client_name": header.get("client_name", ""),
        "tickrate": header.get("tickrate", ""),
        "playback_ticks": header.get("playback_ticks", ""),
        "playback_time": header.get("playback_time", ""),
    }

    # --- Tick data (one file per match) ---
    ticks_df = transformed.get("ticks", pl.DataFrame())
    if isinstance(ticks_df, pl.DataFrame) and not ticks_df.is_empty():
        ticks_dir = output_dir / "ticks"
        ticks_dir.mkdir(exist_ok=True)
        ticks_df.write_parquet(ticks_dir / f"{match_id}.parquet")

    # --- Per-match event tables ---
    for table_name in PER_MATCH_TABLES:
        # Honour skip flags -- don't even write empty files.
        if table_name == "grenades" and config.skip_grenades:
            continue
        if table_name == "damages" and config.skip_damages:
            continue
        if table_name == "shots" and config.skip_shots:
            continue

        df = transformed.get(table_name, pl.DataFrame())
        if not isinstance(df, pl.DataFrame) or df.is_empty():
            continue

        table_dir = output_dir / table_name
        table_dir.mkdir(exist_ok=True)
        df.write_parquet(table_dir / f"{match_id}.parquet")

    return match_row


def append_match_rows(match_rows: list[dict], output_dir: Path) -> None:
    """Write/append rows to the shared matches.parquet file.

    Called once by the parent after all workers have completed.  Because
    this is now O(1) writes (one per run, not one per demo), the old
    align-and-concat pattern is fine.
    """
    if not match_rows:
        return
    new_df = pl.DataFrame(match_rows)
    path = output_dir / "matches.parquet"
    if path.exists():
        existing = pl.read_parquet(path)
        combined = _align_and_concat(existing, new_df)
        # Dedupe by match_id, keeping the newest row.
        combined = combined.unique(subset=["match_id"], keep="last")
        combined.write_parquet(path)
    else:
        new_df.write_parquet(path)


def _align_and_concat(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Concatenate two DataFrames with potentially different schemas.

    Missing columns in either frame are filled with nulls, cast to the type
    from whichever frame already has the column.  Column order follows *a*
    first, then any new columns from *b*.
    """
    all_cols = list(dict.fromkeys(a.columns + b.columns))  # ordered, deduplicated

    # Build a col->dtype map from both frames (a takes precedence).
    dtype_map: dict[str, pl.DataType] = {}
    for col in b.columns:
        dtype_map[col] = b.schema[col]
    for col in a.columns:
        dtype_map[col] = a.schema[col]

    def _pad(df: pl.DataFrame) -> pl.DataFrame:
        for col in all_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype_map[col]).alias(col))
        return df.select(all_cols)

    return pl.concat([_pad(a), _pad(b)])


# ---------------------------------------------------------------------------
# DuckDB registration
# ---------------------------------------------------------------------------


def register_duckdb(output_dir: Path) -> None:
    """Create / update a DuckDB database with views over the Parquet files.

    Views are lightweight pointers -- no data is duplicated.  Per-match
    tables are aggregated via ``read_parquet('<table>/*.parquet')`` globs.
    """
    db_path = output_dir / "cs2_analytics.duckdb"
    con = duckdb.connect(str(db_path))

    try:
        # Shared matches table (single file).
        matches_path = output_dir / "matches.parquet"
        if matches_path.exists():
            con.execute(
                f"CREATE OR REPLACE VIEW matches AS "
                f"SELECT * FROM read_parquet('{matches_path}')"
            )
            log.debug("Registered view 'matches'")

        # Per-match tables (glob of per-match parquet files).
        for table_name in ("ticks", *PER_MATCH_TABLES):
            table_dir = output_dir / table_name
            if not table_dir.exists():
                continue
            files = list(table_dir.glob("*.parquet"))
            if not files:
                continue
            glob_pattern = str(table_dir / "*.parquet")
            con.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_parquet('{glob_pattern}')"
            )
            log.debug(
                "Registered view '%s' (glob, %d files)", table_name, len(files)
            )

        # Convenience: a wide match summary view.
        _create_summary_view(con)

        log.info("DuckDB database updated: %s", db_path)
    finally:
        con.close()


def _create_summary_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create a summary view joining match metadata with aggregate stats."""
    existing_views = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "matches" not in existing_views:
        return

    parts = ["SELECT m.*"]
    joins = []

    if "rounds" in existing_views:
        parts.append("r.n_rounds")
        joins.append(
            "LEFT JOIN (SELECT match_id, COUNT(*) AS n_rounds FROM rounds GROUP BY match_id) r "
            "ON m.match_id = r.match_id"
        )

    if "kills" in existing_views:
        parts.append("k.n_kills")
        joins.append(
            "LEFT JOIN (SELECT match_id, COUNT(*) AS n_kills FROM kills GROUP BY match_id) k "
            "ON m.match_id = k.match_id"
        )

    if "ticks" in existing_views:
        parts.append("t.n_ticks")
        joins.append(
            "LEFT JOIN (SELECT match_id, COUNT(*) AS n_ticks FROM ticks GROUP BY match_id) t "
            "ON m.match_id = t.match_id"
        )

    sql = f"{', '.join(parts)} FROM matches m {' '.join(joins)}"
    con.execute(f"CREATE OR REPLACE VIEW match_summary AS {sql}")


# ---------------------------------------------------------------------------
# Map discovery log
# ---------------------------------------------------------------------------


def append_map_discovery(
    output_dir: Path,
    source_file: str,
    map_name: str,
    match_id: str,
    status: str,
) -> None:
    """Append a single line to the central map discovery log.

    Format:
        {iso_timestamp} {source_file} {map_name} {match_id} {status}

    This lets us know what maps are in which archives without parsing them
    again -- critical for saving time on future runs.
    """
    log_path = output_dir / "map_discovery.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().replace(microsecond=0).isoformat()
    line = f"{ts} {source_file} {map_name or '-'} {match_id or '-'} {status}\n"
    # Called only from the parent process, so no cross-process locking needed.
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating the output directory."""

    ok: bool = True
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def check(self, passed: bool, msg: str) -> None:
        if passed:
            self.checks.append(f"  PASS  {msg}")
        else:
            self.errors.append(f"  FAIL  {msg}")
            self.ok = False

    def warn(self, msg: str) -> None:
        self.warnings.append(f"  WARN  {msg}")

    def print_report(self) -> None:
        log.info("=" * 70)
        log.info("Validation Report")
        log.info("=" * 70)
        for line in self.checks:
            log.info(line)
        for line in self.warnings:
            log.warning(line)
        for line in self.errors:
            log.error(line)
        log.info("-" * 70)
        if self.ok:
            log.info("Result: ALL CHECKS PASSED")
        else:
            log.error("Result: %d CHECK(S) FAILED", len(self.errors))


def validate(output_dir: Path) -> ValidationResult:
    """Validate output directory for completeness and consistency.

    Checks:
    1. Required files exist (matches.parquet, rounds/, ticks/).
    2. Row counts are non-zero where expected.
    3. Schema consistency: match_id present in all tables.
    4. Referential integrity: every match_id in event tables exists in matches.
    5. Tick sampling sanity: no dead players in sampled ticks.
    6. DuckDB database exists and views are queryable.
    """
    vr = ValidationResult()

    # --- 1. File existence ---
    matches_path = output_dir / "matches.parquet"
    rounds_dir = output_dir / "rounds"
    ticks_dir = output_dir / "ticks"
    db_path = output_dir / "cs2_analytics.duckdb"

    vr.check(output_dir.exists(), f"Output directory exists: {output_dir}")
    if not output_dir.exists():
        return vr

    vr.check(matches_path.exists(), "matches.parquet exists")
    vr.check(rounds_dir.exists(), "rounds/ directory exists")
    vr.check(ticks_dir.exists(), "ticks/ directory exists")
    vr.check(db_path.exists(), "cs2_analytics.duckdb exists")

    if not matches_path.exists():
        return vr

    # --- 2. Row counts ---
    matches_df = pl.read_parquet(matches_path)
    n_matches = len(matches_df)
    vr.check(n_matches > 0, f"matches.parquet has {n_matches} row(s)")

    round_files = (
        sorted(rounds_dir.glob("*.parquet")) if rounds_dir.exists() else []
    )
    if round_files:
        n_rounds_total = sum(len(pl.read_parquet(f)) for f in round_files)
        vr.check(
            n_rounds_total > 0,
            f"rounds/ contains {n_rounds_total} row(s) across {len(round_files)} file(s)",
        )
    else:
        vr.warn("rounds/ has no parquet files")

    tick_files = sorted(ticks_dir.glob("*.parquet")) if ticks_dir.exists() else []
    vr.check(len(tick_files) > 0, f"ticks/ contains {len(tick_files)} file(s)")
    vr.check(
        len(tick_files) == n_matches,
        f"Tick file count ({len(tick_files)}) matches demo count ({n_matches})",
    )

    # --- 3. Schema: match_id present ---
    vr.check(
        "match_id" in matches_df.columns, "matches.parquet has 'match_id' column"
    )
    for f in round_files[:3]:  # check first few
        df = pl.read_parquet(f)
        vr.check(
            "match_id" in df.columns,
            f"rounds/{f.name} has 'match_id' column",
        )

    for table_name in PER_MATCH_TABLES:
        table_dir = output_dir / table_name
        if not table_dir.exists():
            vr.warn(f"{table_name}/ not found (may be expected if skipped or empty)")
            continue
        files = list(table_dir.glob("*.parquet"))
        if not files:
            continue
        df = pl.read_parquet(files[0])
        vr.check(
            "match_id" in df.columns,
            f"{table_name}/{files[0].name} has 'match_id' column",
        )

    # --- 4. Referential integrity ---
    known_ids = set(matches_df["match_id"].to_list())

    for table_name in ("rounds", "kills", "damages", "bomb"):
        table_dir = output_dir / table_name
        if not table_dir.exists():
            continue
        orphan_count = 0
        for f in table_dir.glob("*.parquet"):
            df = pl.read_parquet(f)
            if df.is_empty() or "match_id" not in df.columns:
                continue
            table_ids = set(df["match_id"].to_list())
            orphan_count += len(table_ids - known_ids)
        vr.check(
            orphan_count == 0,
            f"{table_name}: all match_ids exist in matches ({orphan_count} orphan(s))",
        )

    for tf in tick_files:
        tick_df = pl.read_parquet(tf)
        if tick_df.is_empty() or "match_id" not in tick_df.columns:
            continue
        tick_ids = set(tick_df["match_id"].to_list())
        orphans = tick_ids - known_ids
        vr.check(
            len(orphans) == 0,
            f"ticks/{tf.name}: all match_ids exist in matches ({len(orphans)} orphan(s))",
        )

    # --- 5. Tick sampling sanity ---
    for tf in tick_files:
        tick_df = pl.read_parquet(tf)
        if tick_df.is_empty():
            vr.warn(f"ticks/{tf.name} is empty")
            continue
        if "health" in tick_df.columns:
            dead = tick_df.filter(pl.col("health") <= 0)
            vr.check(
                len(dead) == 0,
                f"ticks/{tf.name}: no dead players in sampled ticks ({len(dead)} found)",
            )

    # --- 6. DuckDB views ---
    if db_path.exists():
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            views = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
            vr.check("matches" in views, "DuckDB has 'matches' view")
            vr.check("ticks" in views, "DuckDB has 'ticks' view")
            if "match_summary" in views:
                try:
                    summary = con.execute("SELECT * FROM match_summary").fetchdf()
                    vr.check(
                        len(summary) > 0,
                        f"match_summary view returns {len(summary)} row(s)",
                    )
                except Exception as exc:
                    vr.check(False, f"match_summary view queryable: {exc}")
        finally:
            con.close()

    return vr


# ---------------------------------------------------------------------------
# Worker task (runs in a subprocess for parallel parsing)
# ---------------------------------------------------------------------------


def _process_one_task(task: dict) -> DemoResult:
    """Worker entry point: process one source file (.dem or .rar).

    Each worker runs in its own subprocess and owns its own temp dir so
    there's no cross-worker contention.  The task dict contains:

        source_path: str (absolute path to a .dem or .rar file)
        output_dir:  str (shared output directory)
        config:      PipelineConfig (pickled across the process boundary)
    """
    source_path = Path(task["source_path"])
    output_dir = Path(task["output_dir"])
    config: PipelineConfig = task["config"]

    # Per-worker temp directory (unique).  Used only for .rar extraction.
    temp_root = tempfile.TemporaryDirectory(prefix=f"etl_w{os.getpid()}_")
    try:
        return _process_one_source(
            source_path, output_dir, config, Path(temp_root.name)
        )
    finally:
        temp_root.cleanup()


def _process_one_source(
    source_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    temp_dir: Path,
) -> DemoResult:
    """Dispatch a source file (.dem or .rar) to the appropriate processor."""
    suffix = source_path.suffix.lower()

    if suffix == ".dem":
        return _process_one_dem(source_path, source_path.name, output_dir, config)

    if suffix == ".rar":
        return _process_one_rar(source_path, output_dir, config, temp_dir)

    return DemoResult(
        path=str(source_path),
        match_id="",
        success=False,
        status="failed",
        error=f"Unsupported file type: {suffix}",
        source_file=source_path.name,
    )


def _process_one_rar(
    rar_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    temp_dir: Path,
) -> DemoResult:
    """Process a .rar archive.  Returns a carrier DemoResult whose
    ``_child_results`` attribute holds the per-demo DemoResult objects.

    When the archive contains multiple .dem files (e.g. BO3/BO5) each demo
    is processed sequentially inside the worker; parallelism happens at the
    source-file level instead.
    """
    # Cheap map filter via the archive's member listing (no extraction).
    member_dems = list_rar_dem_files(rar_path)

    if config.map_filter and member_dems:
        relevant_detected = [
            detect_map_from_filename(dem_name) for dem_name in member_dems
        ]
        # Keep the archive if at least one member is either (a) definitely
        # in the filter or (b) unknown-from-filename (fallback -> parse).
        any_relevant = any(
            (m is None) or (m in config.map_filter) for m in relevant_detected
        )
        if not any_relevant:
            return DemoResult(
                path=str(rar_path),
                match_id="",
                map_name="",
                status="filtered",
                error=f"No matching maps in archive (contains: {member_dems})",
                source_file=rar_path.name,
            )

    # Extract all .dem files (we keep things simple: extract the whole archive,
    # then per-demo map filter below will skip files we don't want).
    try:
        extracted = extract_rar(rar_path, temp_dir)
    except RuntimeError as exc:
        return DemoResult(
            path=str(rar_path),
            match_id="",
            success=False,
            status="failed",
            error=str(exc),
            source_file=rar_path.name,
        )

    if not extracted:
        return DemoResult(
            path=str(rar_path),
            match_id="",
            success=False,
            status="failed",
            error="No .dem files in archive",
            source_file=rar_path.name,
        )

    # Process each extracted demo; collect per-demo results.
    child_results: list[DemoResult] = []
    for dem_path in extracted:
        child = _process_one_dem(dem_path, rar_path.name, output_dir, config)
        child_results.append(child)

    # Carrier result: aggregated status for top-level display.
    all_ok = all(c.success or c.status in ("skipped", "filtered") for c in child_results)
    any_fail = any(c.status == "failed" for c in child_results)
    carrier = DemoResult(
        path=str(rar_path),
        match_id="",
        source_file=rar_path.name,
        success=all_ok and not any_fail,
        status="failed" if any_fail else "ok",
    )
    # Attach via dynamic attribute for the parent to unpack.
    carrier._child_results = child_results  # type: ignore[attr-defined]
    return carrier


def _process_one_dem(
    dem_path: Path,
    source_file: str,
    output_dir: Path,
    config: PipelineConfig,
) -> DemoResult:
    """Run the full Extract-Transform-Load pipeline for a single .dem file.

    ``source_file`` is the *original* source file name for the map
    discovery log (e.g. ``foo-vs-bar.rar`` for archived demos, or the
    .dem file name for loose .dem inputs).
    """
    mid = match_id_from_path(dem_path)
    result = DemoResult(
        path=str(dem_path),
        match_id=mid,
        source_file=source_file,
    )
    t0 = time.perf_counter()

    # Skip known-bad demos early.
    if dem_path.name in KNOWN_BAD_DEMOS:
        result.error = "Skipped (known bad demo)"
        result.status = "skipped"
        result.elapsed_secs = time.perf_counter() - t0
        return result

    # --- Cheap map filter based on filename (saves full parse) ---
    if config.map_filter:
        detected = detect_map_from_filename(dem_path.name)
        if detected is not None and detected not in config.map_filter:
            result.map_name = detected
            result.status = "filtered"
            result.error = f"Map {detected} not in filter"
            result.elapsed_secs = time.perf_counter() - t0
            return result

    # Skip already-processed demos (idempotent: check for tick file).
    tick_file = output_dir / "ticks" / f"{mid}.parquet"
    if tick_file.exists():
        log.info("  Skipping %s (already processed, match_id=%s)", dem_path.name, mid)
        result.success = True
        result.status = "skipped"
        result.error = "skipped (already exists)"
        try:
            existing_ticks = pl.read_parquet(tick_file)
            result.n_ticks_sampled = len(existing_ticks)
        except Exception:
            pass
        result.elapsed_secs = time.perf_counter() - t0
        return result

    try:
        # --- Extract ---
        log.info("  Parsing %s ...", dem_path.name)
        raw = parse_demo(dem_path, config)
        result.map_name = raw["header"].get("map_name", "")
        result.n_ticks_raw = len(raw.get("ticks", pl.DataFrame()))

        # --- Recheck map filter against parsed header (authoritative) ---
        if config.map_filter and result.map_name not in config.map_filter:
            result.status = "filtered"
            result.error = f"Map {result.map_name} not in filter"
            result.elapsed_secs = time.perf_counter() - t0
            return result

        # --- Transform ---
        log.info("  Transforming (match_id=%s) ...", mid)
        transformed = transform_demo(raw, mid, config.tick_interval)

        ticks_df = transformed.get("ticks", pl.DataFrame())
        result.n_ticks_sampled = (
            len(ticks_df) if isinstance(ticks_df, pl.DataFrame) else 0
        )

        rounds_df = transformed.get("rounds", pl.DataFrame())
        result.n_rounds = (
            len(rounds_df) if isinstance(rounds_df, pl.DataFrame) else 0
        )

        kills_df = transformed.get("kills", pl.DataFrame())
        result.n_kills = len(kills_df) if isinstance(kills_df, pl.DataFrame) else 0

        damages_df = transformed.get("damages", pl.DataFrame())
        result.n_damages = (
            len(damages_df) if isinstance(damages_df, pl.DataFrame) else 0
        )

        bomb_df = transformed.get("bomb", pl.DataFrame())
        result.n_bomb_events = (
            len(bomb_df) if isinstance(bomb_df, pl.DataFrame) else 0
        )

        # --- Load ---
        log.info("  Writing Parquet files ...")
        match_row = write_parquet(transformed, output_dir, mid, config)
        result.match_row = match_row

        result.success = True
        result.status = "ok"

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.status = "failed"
        log.error("  FAILED %s -- %s", dem_path.name, result.error)

    result.elapsed_secs = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Main: orchestrate workers
# ---------------------------------------------------------------------------


def _flatten_results(raw_results: list[DemoResult]) -> list[DemoResult]:
    """Flatten worker results, unpacking .rar carrier results into children."""
    out: list[DemoResult] = []
    for r in raw_results:
        children = getattr(r, "_child_results", None)
        if children:
            out.extend(children)
        else:
            out.append(r)
    return out


def run_pipeline(
    input_files: list[Path],
    output_dir: Path,
    config: PipelineConfig,
    workers: int,
    max_demos: int | None,
) -> PipelineReport:
    """Dispatch the input files to worker processes and collect results.

    Each source file (.dem or .rar) is processed as a single worker task.
    For .rar archives containing multiple .dem files, the worker iterates
    internally so that task-level parallelism stays balanced at the source
    level.

    Results are streamed back via ``imap_unordered`` so that ``--max-demos``
    can short-circuit as soon as the cap is reached.
    """
    report = PipelineReport(start_time=time.perf_counter())
    if not input_files:
        report.end_time = time.perf_counter()
        return report

    tasks = [
        {
            "source_path": str(f),
            "output_dir": str(output_dir),
            "config": config,
        }
        for f in input_files
    ]

    n_successful = 0
    all_match_rows: list[dict] = []

    def _on_result(raw: DemoResult) -> bool:
        """Handle one worker result.  Returns True if we should keep going."""
        nonlocal n_successful
        children = _flatten_results([raw])
        for c in children:
            report.results.append(c)
            if c.match_row is not None:
                all_match_rows.append(c.match_row)
            # Log discovery (one line per demo, covers ok/failed/filtered).
            append_map_discovery(
                output_dir,
                c.source_file or Path(c.path).name,
                c.map_name,
                c.match_id,
                c.status,
            )
            if c.success and c.status == "ok":
                n_successful += 1

            status_upper = c.status.upper()
            log.info(
                "  [%s] %s  map=%s  rounds=%d  ticks=%d  kills=%d  (%.1fs)",
                status_upper,
                Path(c.path).name,
                c.map_name,
                c.n_rounds,
                c.n_ticks_sampled,
                c.n_kills,
                c.elapsed_secs,
            )
        if max_demos is not None and n_successful >= max_demos:
            return False
        return True

    if workers <= 1:
        for task in tqdm(tasks, desc="Processing demos", unit="demo"):
            raw = _process_one_task(task)
            if not _on_result(raw):
                break
    else:
        # Use spawn context to avoid inheriting state from the parent,
        # which matters because awpy / demoparser2 can hold file handles.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            it = pool.imap_unordered(_process_one_task, tasks)
            with tqdm(total=len(tasks), desc="Processing demos", unit="demo") as pbar:
                for raw in it:
                    pbar.update(1)
                    if not _on_result(raw):
                        pool.terminate()
                        break

    # Write shared matches.parquet once, at the end.
    if all_match_rows:
        append_match_rows(all_match_rows, output_dir)

    report.end_time = time.perf_counter()
    return report


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ETL pipeline."""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline: Parse CS2 demos -> Parquet -> DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        nargs="+",
        metavar="PATH",
        help="Input .dem / .rar files or a directory (shell globs work)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for Parquet files and DuckDB",
    )
    parser.add_argument(
        "--tick-rate",
        type=int,
        default=TICK_SAMPLE_INTERVAL,
        help=f"Sample every Nth tick (default: {TICK_SAMPLE_INTERVAL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--map-filter",
        type=str,
        default=None,
        help="Comma-separated map names to include (e.g. 'de_dust2,de_mirage')",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Stop after N successful parses (useful for PoC runs)",
    )
    parser.add_argument(
        "--skip-grenades",
        action="store_true",
        help="Do not parse/store grenade trajectories (saves ~20%% parse time)",
    )
    parser.add_argument(
        "--skip-damages",
        action="store_true",
        help="Do not parse/store damages (player_hurt events)",
    )
    parser.add_argument(
        "--skip-shots",
        action="store_true",
        help="Do not parse/store shots (weapon_fire events)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Shorthand for --skip-grenades --skip-damages --skip-shots",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing output directory and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Validate-only mode ---
    if args.validate:
        vr = validate(args.output)
        vr.print_report()
        return 0 if vr.ok else 1

    # --- Normal ETL mode ---
    if not args.input:
        parser.error("--input is required when not using --validate")

    # Normalize skip flags.
    if args.minimal:
        args.skip_grenades = True
        args.skip_damages = True
        args.skip_shots = True

    # Parse & canonicalize the map filter.
    map_filter: frozenset[str] = frozenset()
    if args.map_filter:
        raw_filter = args.map_filter.replace(" ", "")
        tokens = [m for m in raw_filter.split(",") if m]
        canon: set[str] = set()
        for m in tokens:
            canon.add(m if m.startswith("de_") else f"de_{m}")
        map_filter = frozenset(canon)
        log.info("Map filter: %s", sorted(map_filter))

    config = PipelineConfig(
        tick_interval=args.tick_rate,
        skip_grenades=args.skip_grenades,
        skip_damages=args.skip_damages,
        skip_shots=args.skip_shots,
        map_filter=map_filter,
    )

    input_files = resolve_input_files(args.input)
    if not input_files:
        log.error("No input files matched the given patterns")
        return 1

    # Separate .dem and .rar (informational).
    n_dem = sum(1 for f in input_files if f.suffix.lower() == ".dem")
    n_rar = sum(1 for f in input_files if f.suffix.lower() == ".rar")
    other = [f for f in input_files if f.suffix.lower() not in (".dem", ".rar")]
    if other:
        log.warning("Ignoring %d unsupported file(s)", len(other))
        input_files = [
            f for f in input_files if f.suffix.lower() in (".dem", ".rar")
        ]

    log.info(
        "Input: %d .dem file(s), %d .rar archive(s)  |  workers=%d  tick_interval=%d",
        n_dem,
        n_rar,
        args.workers,
        args.tick_rate,
    )
    log.info(
        "Config: skip_grenades=%s skip_damages=%s skip_shots=%s map_filter=%s max_demos=%s",
        config.skip_grenades,
        config.skip_damages,
        config.skip_shots,
        sorted(map_filter) if map_filter else "-",
        args.max_demos,
    )
    log.info("-" * 70)

    args.output.mkdir(parents=True, exist_ok=True)

    report = run_pipeline(
        input_files=input_files,
        output_dir=args.output,
        config=config,
        workers=max(1, args.workers),
        max_demos=args.max_demos,
    )

    # --- Register in DuckDB ---
    if report.succeeded:
        log.info("Registering views in DuckDB ...")
        try:
            register_duckdb(args.output)
        except Exception as exc:
            log.error("DuckDB registration failed: %s", exc)

    report.print_summary()

    # --- Run validation ---
    if report.succeeded:
        log.info("")
        log.info("Running post-pipeline validation ...")
        vr = validate(args.output)
        vr.print_report()

    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
