#!/usr/bin/env python3
"""
Tick-field reliability analysis for CS2 pro demos.
Part A: Re-parse 12 demos with full 29-col schema, check new fields.
Part B: Query existing DuckDB (100 demos) for existing field reliability.
"""

from __future__ import annotations

import pathlib
import rarfile
import tempfile
import traceback
import warnings
from collections import defaultdict

import duckdb
import polars as pl
from awpy import Demo

warnings.filterwarnings("ignore")

RAR_DIR = pathlib.Path("/mnt/d/projects/fps/misc_cs/hltv-scrape/data")
DUCKDB_PATH = pathlib.Path("/mnt/d/projects/fps/awpy/csgo/output/full_dust2/cs2_analytics.duckdb")

EXTRA_PLAYER_PROPS = [
    "armor_value",
    "current_equip_value",
    "has_defuser",
    "has_helmet",
    "weapon_name",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "yaw",
    "pitch",
    "flash_duration",
    "zoom_lvl",
    "is_walking",
    "duck_amount",
    "spotted",
    "is_bomb_planted",
    "which_bomb_zone",
    "game_time",
]

# 12 demos: 5 dust2, 3 inferno, 2 nuke, 2 mirage
TARGET_RARS = [
    "blasttv-austin-major-2025-aurora-vs-furia-dust2-xQn4_tD4kMLwrLXWK3KvF0.rar",
    "blasttv-austin-major-2025-g2-vs-pain-dust2-lf4ZZjZU_uqIY9M5_jzycB.rar",
    "blasttv-austin-major-2025-liquid-vs-lynn-vision-dust2-F4utTQoPumQMNK8ZYCX04u.rar",
    "blasttv-austin-major-2025-spirit-vs-lynn-vision-dust2-iH45g5AQaBjEUbSg160o2P.rar",
    "blasttv-austin-major-2025-spirit-vs-pain-dust2-e3Myt4k8lU9B0YdKb7KAzT.rar",
    "blasttv-austin-major-2025-3dmax-vs-natus-vincere-inferno-fYqJGUSLlnbSjPnBTGEuXX.rar",
    "blasttv-austin-major-2025-aurora-vs-g2-inferno-f38KvSWe_bUAFtpFu5oIVa.rar",
    "blasttv-austin-major-2025-stage-1-complexity-vs-fluxo-inferno-RERzarczIW5QIhRElUmkwA.rar",
    "blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb.rar",
    "blasttv-austin-major-2025-europe-regional-qualifier-astralis-vs-og-nuke-8btcvuX9bzdNwnywC8Q58g.rar",
    "blasttv-austin-major-2025-europe-regional-qualifier-astralis-vs-parivision-mirage-Wkz7W9cYMPRbaRjkqkMk2e.rar",
    "blasttv-austin-major-2025-europe-regional-qualifier-gamerlegion-vs-nemiga-mirage-hifTA7FeMyhZOyEvdkHfos.rar",
]

CS2_KNOWN_WEAPONS = {
    # Rifles
    "AK-47", "M4A1-S", "M4A4", "SG 553", "AUG", "FAMAS", "Galil AR",
    "SSG 08", "AWP", "G3SG1", "SCAR-20",
    # Pistols
    "Glock-18", "USP-S", "P2000", "Dual Berettas", "P250", "Five-SeveN",
    "Tec-9", "CZ75-Auto", "Desert Eagle", "R8 Revolver",
    # SMGs
    "MAC-10", "MP9", "MP5-SD", "UMP-45", "PP-Bizon", "P90", "MP7",
    # Heavy
    "Nova", "XM1014", "Sawed-Off", "MAG-7", "M249", "Negev",
    # Knives/melee
    "Knife", "Bayonet", "Flip Knife", "Gut Knife", "Karambit",
    "M9 Bayonet", "Huntsman Knife", "Falchion Knife", "Bowie Knife",
    "Butterfly Knife", "Shadow Daggers", "Navaja Knife", "Stiletto Knife",
    "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife",
    "Nomad Knife", "Skeleton Knife", "Ursus Knife", "Cord Knife",
    # Equipment / nades
    "HE Grenade", "Flashbang", "Smoke Grenade", "Molotov", "Incendiary Grenade",
    "Decoy Grenade", "C4", "Zeus x27", "Defuse Kit",
}

# Approximate AWP cost used in cross-validation
AWP_COST = 4750
ECO_THRESHOLD = 500

results_by_demo: list[dict] = []


def parse_rar(rar_path: pathlib.Path) -> pl.DataFrame | None:
    try:
        with rarfile.RarFile(rar_path) as rf:
            dem_files = [f for f in rf.namelist() if f.endswith(".dem")]
            if not dem_files:
                print(f"  [SKIP] No .dem in {rar_path.name}")
                return None
            dem_name = dem_files[0]
            with tempfile.TemporaryDirectory() as tmpdir:
                rf.extract(dem_name, tmpdir)
                dem_path = pathlib.Path(tmpdir) / dem_name
                demo = Demo(path=dem_path)
                demo.parse(player_props=EXTRA_PLAYER_PROPS)
                ticks = demo.ticks
                if ticks is None or ticks.is_empty():
                    print(f"  [SKIP] Empty ticks in {rar_path.name}")
                    return None
                return ticks
    except Exception as e:
        print(f"  [ERROR] {rar_path.name}: {e}")
        traceback.print_exc()
        return None


def null_rate(df: pl.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return df[col].is_null().mean()


def value_counts_top(df: pl.DataFrame, col: str, n: int = 15) -> list[tuple]:
    if col not in df.columns:
        return []
    vc = df[col].value_counts(sort=True).head(n)
    return list(zip(vc[col].to_list(), vc["count"].to_list()))


def analyze_demo(ticks: pl.DataFrame, demo_name: str) -> dict:
    n = len(ticks)
    result = {
        "demo": demo_name,
        "n_rows": n,
        "errors": [],
    }

    # --- NULL RATES for all new + key existing fields ---
    fields_to_check = [
        "weapon_name", "pitch", "is_walking", "duck_amount", "spotted",
        # existing
        "zoom_lvl", "place", "current_equip_value", "velocity_X", "velocity_Y", "velocity_Z",
        "yaw", "flash_duration", "is_bomb_planted", "which_bomb_zone", "game_time",
        "health", "armor", "X", "Y", "Z",
    ]
    for f in fields_to_check:
        result[f"null_{f}"] = null_rate(ticks, f)

    # --- weapon_name value_counts ---
    if "weapon_name" in ticks.columns:
        vc = value_counts_top(ticks, "weapon_name", 20)
        result["weapon_name_vc"] = vc
        weapon_vals = set(ticks["weapon_name"].drop_nulls().unique().to_list())
        unknown = weapon_vals - CS2_KNOWN_WEAPONS
        result["weapon_name_unknown"] = sorted(unknown)
        result["weapon_name_unique_count"] = len(weapon_vals)

        # Check for empty strings or integers
        has_empty = ticks.filter(pl.col("weapon_name") == "").height > 0
        result["weapon_name_has_empty_str"] = has_empty
    else:
        result["weapon_name_vc"] = []
        result["weapon_name_unknown"] = []
        result["weapon_name_unique_count"] = 0
        result["weapon_name_has_empty_str"] = False

    # --- Cross-validate weapon_name vs current_equip_value ---
    if "weapon_name" in ticks.columns and "current_equip_value" in ticks.columns:
        # AWP holders: equip_value should be >= AWP_COST
        awp_rows = ticks.filter(pl.col("weapon_name") == "AWP").filter(
            pl.col("health") > 0
        )
        if awp_rows.height > 0:
            awp_below_cost = awp_rows.filter(
                pl.col("current_equip_value").is_not_null()
                & (pl.col("current_equip_value") < AWP_COST)
            ).height
            result["awp_below_cost_count"] = awp_below_cost
            result["awp_total_count"] = awp_rows.height
            result["awp_below_cost_pct"] = awp_below_cost / awp_rows.height
        else:
            result["awp_below_cost_count"] = 0
            result["awp_total_count"] = 0
            result["awp_below_cost_pct"] = 0.0

        # Eco players (equip < ECO_THRESHOLD): should only have pistols/knives
        eco_rows = ticks.filter(
            pl.col("current_equip_value").is_not_null()
            & (pl.col("current_equip_value") < ECO_THRESHOLD)
            & (pl.col("health") > 0)
        )
        if eco_rows.height > 0:
            eco_weapons_vc = value_counts_top(eco_rows, "weapon_name", 10)
            result["eco_weapons_vc"] = eco_weapons_vc
            # Flag rifles in eco
            rifle_names = {"AK-47", "M4A1-S", "M4A4", "SG 553", "AUG", "FAMAS", "Galil AR", "AWP"}
            if "weapon_name" in eco_rows.columns:
                eco_rifles = eco_rows.filter(
                    pl.col("weapon_name").is_in(list(rifle_names))
                ).height
                result["eco_rifles_count"] = eco_rifles
            else:
                result["eco_rifles_count"] = 0
        else:
            result["eco_weapons_vc"] = []
            result["eco_rifles_count"] = 0

    # --- zoom_lvl vs weapon_name cross-validation ---
    if "zoom_lvl" in ticks.columns and "weapon_name" in ticks.columns:
        # AWP should have zoom_lvl 1 or 2
        awp_alive = ticks.filter(
            (pl.col("weapon_name") == "AWP") & (pl.col("health") > 0)
        )
        if awp_alive.height > 0:
            awp_zoom_vc = value_counts_top(awp_alive, "zoom_lvl", 5)
            result["awp_zoom_vc"] = awp_zoom_vc

        # Rifle players should have zoom_lvl = 0 or null
        rifles = {"AK-47", "M4A1-S", "M4A4", "SG 553", "AUG", "FAMAS", "Galil AR"}
        rifle_rows = ticks.filter(
            pl.col("weapon_name").is_in(list(rifles)) & (pl.col("health") > 0)
        )
        if rifle_rows.height > 0:
            rifle_zoomed = rifle_rows.filter(
                pl.col("zoom_lvl").is_not_null() & (pl.col("zoom_lvl") > 0)
            ).height
            result["rifle_nonzero_zoom"] = rifle_zoomed
            result["rifle_total"] = rifle_rows.height
        else:
            result["rifle_nonzero_zoom"] = 0
            result["rifle_total"] = 0

    # --- spotted sanity: dead players (health=0) should never be spotted=True ---
    if "spotted" in ticks.columns and "health" in ticks.columns:
        dead_spotted = ticks.filter(
            (pl.col("health") == 0) & (pl.col("spotted") == True)
        ).height
        dead_total = ticks.filter(pl.col("health") == 0).height
        result["dead_and_spotted"] = dead_spotted
        result["dead_total"] = dead_total

    # --- pitch distribution ---
    if "pitch" in ticks.columns:
        pitch_vals = ticks["pitch"].drop_nulls()
        if len(pitch_vals) > 0:
            result["pitch_mean"] = pitch_vals.mean()
            result["pitch_std"] = pitch_vals.std()
            result["pitch_min"] = pitch_vals.min()
            result["pitch_max"] = pitch_vals.max()
            # Count out-of-range [-90, 90]
            out_of_range = ticks.filter(
                pl.col("pitch").is_not_null()
                & ((pl.col("pitch") < -90) | (pl.col("pitch") > 90))
            ).height
            result["pitch_out_of_range"] = out_of_range

    # --- duck_amount distribution ---
    if "duck_amount" in ticks.columns:
        duck_vals = ticks["duck_amount"].drop_nulls()
        if len(duck_vals) > 0:
            result["duck_mean"] = duck_vals.mean()
            result["duck_std"] = duck_vals.std()
            # standing (<=0.05)
            result["duck_standing_pct"] = (duck_vals <= 0.05).mean()
            # crouched (>=0.95)
            result["duck_crouched_pct"] = (duck_vals >= 0.95).mean()
            # mid-crouch (0.05 < x < 0.95)
            mid = ((duck_vals > 0.05) & (duck_vals < 0.95)).mean()
            result["duck_mid_pct"] = float(mid)

    # --- is_walking rate ---
    if "is_walking" in ticks.columns:
        walk_vals = ticks["is_walking"].drop_nulls()
        if len(walk_vals) > 0:
            result["is_walking_rate"] = float((walk_vals == True).mean())
        else:
            result["is_walking_rate"] = float("nan")

    # --- current_equip_value distribution ---
    if "current_equip_value" in ticks.columns:
        eq = ticks["current_equip_value"].drop_nulls()
        if len(eq) > 0:
            result["equip_min"] = eq.min()
            result["equip_max"] = eq.max()
            result["equip_mean"] = eq.mean()

    return result


# ============================================================
# PART A: Parse 12 fresh demos
# ============================================================
print("=" * 70)
print("PART A: Parsing 12 demos with full 29-column schema")
print("=" * 70)

all_ticks: list[pl.DataFrame] = []
demo_results: list[dict] = []

for rar_name in TARGET_RARS:
    rar_path = RAR_DIR / rar_name
    if not rar_path.exists():
        print(f"  [MISSING] {rar_name}")
        continue
    demo_short = rar_name.split("-")[4:7]  # extract map + teams
    demo_label = "-".join(rar_name.split("-")[3:6])
    print(f"\nParsing: {demo_label}...")
    ticks = parse_rar(rar_path)
    if ticks is None:
        continue
    print(f"  -> {len(ticks):,} rows, columns={ticks.columns[:5]}...")
    r = analyze_demo(ticks, demo_label)
    demo_results.append(r)
    all_ticks.append(ticks)

print(f"\n\nSuccessfully parsed {len(demo_results)} demos.")

# ============================================================
# SUMMARY TABLE A1: Null rates per demo per field
# ============================================================
new_fields = ["weapon_name", "pitch", "is_walking", "duck_amount", "spotted"]
existing_fields = ["zoom_lvl", "place", "current_equip_value", "velocity_X", "velocity_Y", "velocity_Z"]

print("\n" + "=" * 70)
print("TABLE A1: NULL RATES (%) per field per demo")
print("=" * 70)

all_check_fields = new_fields + existing_fields
header = f"{'Demo':<45}" + "".join(f"{f[:10]:<12}" for f in all_check_fields)
print(header)
print("-" * len(header))

for r in demo_results:
    row = f"{r['demo'][:44]:<45}"
    for f in all_check_fields:
        v = r.get(f"null_{f}", float("nan"))
        if v != v:  # nan
            row += f"{'N/A':<12}"
        elif v > 0.05:
            row += f"{'!'+f'{v*100:.1f}%':<12}"  # flag >5%
        else:
            row += f"{v*100:.2f}%    "
    print(row)

# ============================================================
# TABLE A2: weapon_name value counts (aggregate across all demos)
# ============================================================
print("\n" + "=" * 70)
print("TABLE A2: weapon_name VALUE COUNTS (aggregate, top 25)")
print("=" * 70)

if all_ticks:
    combined = pl.concat(all_ticks, how="diagonal_relaxed")
    total_rows = len(combined)
    print(f"Total rows across {len(all_ticks)} demos: {total_rows:,}")

    wn_col = "weapon_name"
    if wn_col in combined.columns:
        vc = combined[wn_col].value_counts(sort=True).head(25)
        for row in vc.iter_rows():
            name, cnt = row[0], row[1]
            pct = cnt / total_rows * 100
            flag = " [UNKNOWN]" if name and name not in CS2_KNOWN_WEAPONS else ""
            print(f"  {str(name):<30} {cnt:>8,}  ({pct:.2f}%){flag}")

        null_cnt = combined[wn_col].is_null().sum()
        empty_cnt = (combined[wn_col] == "").sum() if combined[wn_col].dtype == pl.Utf8 else 0
        print(f"\n  NULL:                          {null_cnt:>8,}  ({null_cnt/total_rows*100:.2f}%)")
        print(f"  EMPTY STRING:                  {empty_cnt:>8,}  ({empty_cnt/total_rows*100:.2f}%)")

# ============================================================
# TABLE A3: Cross-validation weapon_name vs equip_value
# ============================================================
print("\n" + "=" * 70)
print("TABLE A3: CROSS-VALIDATION weapon_name vs current_equip_value")
print("=" * 70)

print(f"\n{'Demo':<45} {'AWP<$4750':<12} {'AWP_total':<12} {'EcoRifles':<12}")
print("-" * 85)
for r in demo_results:
    awp_pct = r.get("awp_below_cost_pct", 0)
    awp_flag = " !" if awp_pct > 0.1 else ""
    print(
        f"{r['demo'][:44]:<45} "
        f"{awp_pct*100:.1f}%{awp_flag:<8} "
        f"{r.get('awp_total_count', 0):<12} "
        f"{r.get('eco_rifles_count', 0):<12}"
    )

print("\nEco weapons breakdown (equip<500, alive players) - Demo 1:")
if demo_results:
    eco_vc = demo_results[0].get("eco_weapons_vc", [])
    for name, cnt in eco_vc[:10]:
        print(f"  {str(name):<30} {cnt}")

# ============================================================
# TABLE A4: zoom_lvl vs weapon_name
# ============================================================
print("\n" + "=" * 70)
print("TABLE A4: zoom_lvl vs weapon_name (AWP zoom distribution)")
print("=" * 70)

print(f"\n{'Demo':<45} AWP zoom distribution              Rifle nonzero_zoom")
print("-" * 90)
for r in demo_results:
    awp_zoom = r.get("awp_zoom_vc", [])
    awp_zoom_str = str(awp_zoom[:3])
    rifle_nz = r.get("rifle_nonzero_zoom", 0)
    rifle_tot = r.get("rifle_total", 1)
    print(f"{r['demo'][:44]:<45} {awp_zoom_str:<35} {rifle_nz}/{rifle_tot}")

# ============================================================
# TABLE A5: spotted sanity (dead players)
# ============================================================
print("\n" + "=" * 70)
print("TABLE A5: spotted SANITY CHECK (dead players spotted=True?)")
print("=" * 70)

total_dead_spotted = sum(r.get("dead_and_spotted", 0) for r in demo_results)
total_dead = sum(r.get("dead_total", 0) for r in demo_results)
print(f"\nAggregate: dead & spotted={total_dead_spotted:,} / dead_total={total_dead:,}")
for r in demo_results:
    ds = r.get("dead_and_spotted", 0)
    flag = " [!!]" if ds > 0 else ""
    print(f"  {r['demo'][:44]:<45} {ds}{flag}")

# ============================================================
# TABLE A6: pitch distribution
# ============================================================
print("\n" + "=" * 70)
print("TABLE A6: pitch DISTRIBUTION (expected ~N(0, 10°))")
print("=" * 70)

print(f"\n{'Demo':<45} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'OOR':>6}")
print("-" * 95)
for r in demo_results:
    print(
        f"{r['demo'][:44]:<45} "
        f"{r.get('pitch_mean', float('nan')):8.2f} "
        f"{r.get('pitch_std', float('nan')):8.2f} "
        f"{r.get('pitch_min', float('nan')):8.2f} "
        f"{r.get('pitch_max', float('nan')):8.2f} "
        f"{r.get('pitch_out_of_range', 0):6}"
    )

# ============================================================
# TABLE A7: duck_amount distribution
# ============================================================
print("\n" + "=" * 70)
print("TABLE A7: duck_amount DISTRIBUTION (bimodal expected: 0=stand, 1=crouch)")
print("=" * 70)

print(f"\n{'Demo':<45} {'mean':>7} {'std':>7} {'stand%':>8} {'crouch%':>8} {'mid%':>7}")
print("-" * 95)
for r in demo_results:
    print(
        f"{r['demo'][:44]:<45} "
        f"{r.get('duck_mean', float('nan')):7.3f} "
        f"{r.get('duck_std', float('nan')):7.3f} "
        f"{r.get('duck_standing_pct', float('nan'))*100:7.1f}% "
        f"{r.get('duck_crouched_pct', float('nan'))*100:7.1f}% "
        f"{r.get('duck_mid_pct', float('nan'))*100:6.1f}%"
    )

# ============================================================
# TABLE A8: is_walking rate
# ============================================================
print("\n" + "=" * 70)
print("TABLE A8: is_walking RATE (expected 5-15% of live ticks)")
print("=" * 70)

for r in demo_results:
    rate = r.get("is_walking_rate", float("nan"))
    flag = " [LOW?]" if rate == rate and rate < 0.03 else (" [HIGH?]" if rate == rate and rate > 0.20 else "")
    print(f"  {r['demo'][:44]:<45} {rate*100:.2f}%{flag}")

# ============================================================
# PART B: Query existing DuckDB (100 demos)
# ============================================================
print("\n" + "=" * 70)
print("PART B: DuckDB analysis (100 demos, existing 24-col schema)")
print("=" * 70)

if not DUCKDB_PATH.exists():
    print(f"[ERROR] DuckDB not found at {DUCKDB_PATH}")
else:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    # B1: total row count
    total = con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    print(f"\nTotal tick rows in DuckDB: {total:,}")

    # B2: zoom_lvl distribution
    print("\n--- B1: zoom_lvl distribution ---")
    try:
        zoom_q = con.execute("""
            SELECT
                zoom_lvl,
                COUNT(*) as cnt,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
            FROM ticks
            GROUP BY zoom_lvl
            ORDER BY cnt DESC
        """).fetchall()
        for row in zoom_q:
            print(f"  zoom_lvl={row[0]:<6} {row[1]:>10,}  {row[2]:.2f}%")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # B3: zoom_lvl null by weapon class (if weapon_name exists)
    print("\n--- B2: Null rates for key fields across all 100 demos ---")
    key_fields = [
        "zoom_lvl", "place", "current_equip_value",
        "velocity_X", "velocity_Y", "velocity_Z",
        "health", "X", "Y", "Z", "is_bomb_planted",
    ]
    for f in key_fields:
        try:
            r = con.execute(f"""
                SELECT
                    SUM(CASE WHEN {f} IS NULL THEN 1 ELSE 0 END) as nulls,
                    COUNT(*) as total,
                    ROUND(SUM(CASE WHEN {f} IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) as null_pct
                FROM ticks
            """).fetchone()
            flag = " [!!]" if r[2] > 5 else ("")
            print(f"  {f:<25} nulls={r[0]:>10,}  total={r[1]:>12,}  null%={r[2]:.4f}%{flag}")
        except Exception as e:
            print(f"  {f:<25} [MISSING in schema or error: {e}]")

    # B4: zoom_lvl nulls — which ticks? Look at health
    print("\n--- B3: zoom_lvl null breakdown by health ---")
    try:
        r = con.execute("""
            SELECT
                CASE WHEN health = 0 THEN 'dead' ELSE 'alive' END as life_state,
                zoom_lvl IS NULL as is_null,
                COUNT(*) as cnt
            FROM ticks
            GROUP BY life_state, is_null
            ORDER BY life_state, is_null
        """).fetchall()
        for row in r:
            print(f"  {row[0]:<8} zoom_null={row[1]}  {row[2]:>10,}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # B5: place coverage
    print("\n--- B4: place coverage per demo (any demos with 0 place?) ---")
    try:
        place_q = con.execute("""
            SELECT
                match_id,
                COUNT(*) as total,
                SUM(CASE WHEN place IS NULL OR place = '' THEN 1 ELSE 0 END) as null_place,
                ROUND(SUM(CASE WHEN place IS NULL OR place = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as null_pct
            FROM ticks
            GROUP BY match_id
            ORDER BY null_pct DESC
            LIMIT 20
        """).fetchall()
        print(f"  {'match_id':<40} {'total':>10} {'null_place':>12} {'null%':>8}")
        for row in place_q[:20]:
            flag = " [!!]" if row[3] > 5 else ""
            print(f"  {str(row[0])[:39]:<40} {row[1]:>10,} {row[2]:>12,} {row[3]:>7.1f}%{flag}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # B6: current_equip_value distribution
    print("\n--- B5: current_equip_value distribution ---")
    try:
        eq_q = con.execute("""
            SELECT
                MIN(current_equip_value) as min_val,
                MAX(current_equip_value) as max_val,
                AVG(current_equip_value) as avg_val,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY current_equip_value) as median,
                COUNT(CASE WHEN current_equip_value < 0 THEN 1 END) as neg_count,
                COUNT(CASE WHEN current_equip_value > 50000 THEN 1 END) as very_high_count
            FROM ticks
            WHERE current_equip_value IS NOT NULL
        """).fetchone()
        print(f"  min={eq_q[0]:,}  max={eq_q[1]:,}  avg={eq_q[2]:.0f}  median={eq_q[3]:.0f}")
        print(f"  negative_values={eq_q[4]}  extremely_high(>50k)={eq_q[5]}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # B7: velocity null investigation
    print("\n--- B6: velocity null rows — what state are they in? ---")
    try:
        vel_q = con.execute("""
            SELECT
                CASE WHEN health = 0 THEN 'dead' ELSE 'alive' END as life_state,
                COUNT(*) as null_vel_rows
            FROM ticks
            WHERE velocity_X IS NULL
            GROUP BY life_state
        """).fetchall()
        for row in vel_q:
            print(f"  velocity_X null, {row[0]}: {row[1]:,} rows")
    except Exception as e:
        print(f"  [ERROR] {e}")

    con.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
