#!/usr/bin/env python3
"""
Follow-up reliability queries:
1. weapon_name=None rows — are they always dead players?
2. rifle nonzero zoom: which rifles, which zoom values?
3. spotted=True on dead: is it a timing artifact (first tick of death)?
4. is_walking rates — include only alive ticks for fair comparison
5. zoom_lvl null breakdown — where do the 40% nulls come from?
6. HE Grenade / C4 Explosive naming vs known set
7. weapon_name cross-demo consistency check
"""

from __future__ import annotations
import pathlib, rarfile, tempfile, traceback, warnings
import polars as pl
from awpy import Demo

warnings.filterwarnings("ignore")

RAR_DIR = pathlib.Path("/mnt/d/projects/fps/misc_cs/hltv-scrape/data")

EXTRA_PLAYER_PROPS = [
    "armor_value", "current_equip_value", "has_defuser", "has_helmet",
    "weapon_name", "velocity_X", "velocity_Y", "velocity_Z", "yaw", "pitch",
    "flash_duration", "zoom_lvl", "is_walking", "duck_amount", "spotted",
    "is_bomb_planted", "which_bomb_zone", "game_time",
]

def parse_rar(rar_path):
    with rarfile.RarFile(rar_path) as rf:
        dem_files = [f for f in rf.namelist() if f.endswith(".dem")]
        with tempfile.TemporaryDirectory() as tmpdir:
            rf.extract(dem_files[0], tmpdir)
            dem_path = pathlib.Path(tmpdir) / dem_files[0]
            demo = Demo(path=dem_path)
            demo.parse(player_props=EXTRA_PLAYER_PROPS)
            return demo.ticks

# Use first dust2 demo for deep-dive
DEMO_RAR = "blasttv-austin-major-2025-aurora-vs-furia-dust2-xQn4_tD4kMLwrLXWK3KvF0.rar"
DEMO2_RAR = "blasttv-austin-major-2025-g2-vs-pain-dust2-lf4ZZjZU_uqIY9M5_jzycB.rar"
# mirage (one of the two that failed - swap for a working one)
DEMO3_RAR = "blasttv-austin-major-2025-mouz-vs-virtuspro-mirage-A6msyu7GKsphWdb8XK9jR4.rar"

print("Loading demo 1 (aurora vs furia, dust2)...")
df1 = parse_rar(RAR_DIR / DEMO_RAR)
print(f"  {len(df1):,} rows, cols={sorted(df1.columns)}\n")

# ============================================================
# Q1: weapon_name=None — always dead players?
# ============================================================
print("=" * 60)
print("Q1: weapon_name=None root cause")
print("=" * 60)

null_wn = df1.filter(pl.col("weapon_name").is_null())
alive_null_wn = null_wn.filter(pl.col("health") > 0).height
dead_null_wn = null_wn.filter(pl.col("health") == 0).height
total_null_wn = null_wn.height
print(f"weapon_name=None total: {total_null_wn:,}")
print(f"  of which alive (health>0): {alive_null_wn:,}  ({alive_null_wn/total_null_wn*100:.1f}%)")
print(f"  of which dead  (health=0): {dead_null_wn:,}  ({dead_null_wn/total_null_wn*100:.1f}%)")

# Among alive players with null weapon_name, what do other fields show?
if alive_null_wn > 0:
    alive_null_sample = null_wn.filter(pl.col("health") > 0).head(5)
    print(f"\nSample alive rows with null weapon_name:")
    cols = ["health", "current_equip_value", "zoom_lvl", "is_walking", "duck_amount", "spotted"]
    print(alive_null_sample.select([c for c in cols if c in alive_null_sample.columns]))

# Check: null weapon_name vs total — by health bucket
print("\nweapon_name null rate by health:")
buckets = [
    ("dead (h=0)", pl.col("health") == 0),
    ("low hp (1-10)", (pl.col("health") >= 1) & (pl.col("health") <= 10)),
    ("alive (h>10)", pl.col("health") > 10),
]
for label, cond in buckets:
    sub = df1.filter(cond)
    null_r = sub["weapon_name"].is_null().mean() if "weapon_name" in sub.columns else float("nan")
    print(f"  {label:<20}: {null_r*100:.1f}% null  (n={len(sub):,})")

# ============================================================
# Q2: rifle nonzero zoom — which rifles? demo g2 vs pain had 271
# ============================================================
print("\n" + "=" * 60)
print("Q2: rifle nonzero zoom_lvl investigation")
print("=" * 60)

print("Loading demo 2 (g2 vs pain, dust2)...")
df2 = parse_rar(RAR_DIR / DEMO2_RAR)
print(f"  {len(df2):,} rows\n")

rifles = {"AK-47", "M4A1-S", "M4A4", "SG 553", "AUG", "FAMAS", "Galil AR"}
rifle_zoomed = df2.filter(
    pl.col("weapon_name").is_in(list(rifles))
    & pl.col("zoom_lvl").is_not_null()
    & (pl.col("zoom_lvl") > 0)
)
print(f"Rifle rows with zoom_lvl > 0: {rifle_zoomed.height}")
if rifle_zoomed.height > 0:
    print("Breakdown by weapon + zoom_lvl:")
    print(
        rifle_zoomed.group_by(["weapon_name", "zoom_lvl"])
        .agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True)
        .head(10)
    )

# SG 553 and AUG can legitimately zoom — check for those
print("\nNote: SG 553 and AUG have scopes (zoom_lvl=1 is valid for them)")
sg_aug_zoom = df2.filter(
    pl.col("weapon_name").is_in(["SG 553", "AUG"])
    & pl.col("zoom_lvl").is_not_null()
    & (pl.col("zoom_lvl") > 0)
)
print(f"SG553/AUG zoomed rows: {sg_aug_zoom.height}")
non_scoped_zoomed = df2.filter(
    pl.col("weapon_name").is_in(["AK-47", "M4A1-S", "M4A4", "FAMAS", "Galil AR"])
    & pl.col("zoom_lvl").is_not_null()
    & (pl.col("zoom_lvl") > 0)
)
print(f"Non-scoped rifle zoomed rows (AK/M4/Galil/FAMAS): {non_scoped_zoomed.height}")

# Europe-regional astralis vs og nuke also had 1354 - check
DEMO4_RAR = "blasttv-austin-major-2025-europe-regional-qualifier-astralis-vs-og-nuke-8btcvuX9bzdNwnywC8Q58g.rar"
print("\nLoading demo 4 (astralis vs og, nuke)...")
df4 = parse_rar(RAR_DIR / DEMO4_RAR)
print(f"  {len(df4):,} rows")
non_scoped_z4 = df4.filter(
    pl.col("weapon_name").is_in(["AK-47", "M4A1-S", "M4A4", "FAMAS", "Galil AR"])
    & pl.col("zoom_lvl").is_not_null()
    & (pl.col("zoom_lvl") > 0)
)
print(f"Non-scoped rifle zoomed rows (nuke demo): {non_scoped_z4.height}")
if non_scoped_z4.height > 0:
    print(non_scoped_z4.group_by(["weapon_name", "zoom_lvl"]).agg(pl.len().alias("cnt")).sort("cnt", descending=True))

# ============================================================
# Q3: spotted=True on dead — timing artifact?
# ============================================================
print("\n" + "=" * 60)
print("Q3: spotted=True on dead players — timing investigation")
print("=" * 60)

dead_spotted = df1.filter((pl.col("health") == 0) & (pl.col("spotted") == True))
print(f"Dead+spotted rows in demo1: {dead_spotted.height:,}")

# Look at the tick before the dead_spotted rows — did they just die?
# Group by steamid, look at consecutive ticks
if "tick" in df1.columns and "steamid" in df1.columns:
    # For each dead+spotted row, check if there's a tick before with health > 0
    spotted_ticks = dead_spotted.select(["tick", "steamid"]).unique()
    print(f"Unique (tick, steamid) dead+spotted pairs: {spotted_ticks.height:,}")

    # Sample: what spotted value is on the first tick of death?
    # Get deaths: transition from health>0 to health=0
    # Sort df1 by steamid, tick
    df1_sorted = df1.sort(["steamid", "tick"])

    # Shift health to detect death transitions
    df1_with_prev = df1_sorted.with_columns([
        pl.col("health").shift(1).over("steamid").alias("prev_health"),
        pl.col("spotted").shift(1).over("steamid").alias("prev_spotted"),
    ])

    # First tick of death = prev_health > 0, current health = 0
    first_tick_death = df1_with_prev.filter(
        (pl.col("health") == 0)
        & (pl.col("prev_health").is_not_null())
        & (pl.col("prev_health") > 0)
    )
    print(f"\nFirst-tick-of-death rows: {first_tick_death.height}")
    if first_tick_death.height > 0:
        spotted_on_first_tick = first_tick_death["spotted"].value_counts(sort=True)
        print("spotted distribution on FIRST tick of death:")
        print(spotted_on_first_tick)

    # How many death ticks are there total?
    all_death_ticks = df1_with_prev.filter(
        (pl.col("health") == 0)
        & (pl.col("prev_health").is_not_null())
        & (pl.col("prev_health") > 0)
    ).height
    print(f"\n  -> spotted=True on first death tick is {'expected' if first_tick_death.height > 0 else 'unexpected'}")

    # How many subsequent dead ticks also have spotted=True?
    # Check: for rows where prev_health == 0 AND health == 0 (not-first-tick dead)
    subsequent_dead = df1_with_prev.filter(
        (pl.col("health") == 0)
        & (pl.col("prev_health").is_not_null())
        & (pl.col("prev_health") == 0)
    )
    if subsequent_dead.height > 0:
        sub_spotted = subsequent_dead["spotted"].value_counts(sort=True)
        print(f"\nSubsequent dead ticks (prev also dead): {subsequent_dead.height:,}")
        print("spotted distribution on SUBSEQUENT dead ticks:")
        print(sub_spotted)

# ============================================================
# Q4: is_walking — alive-only rate
# ============================================================
print("\n" + "=" * 60)
print("Q4: is_walking rate (alive only vs all ticks)")
print("=" * 60)

for label, df_x in [("demo1 dust2", df1), ("demo2 dust2", df2)]:
    all_rate = (df_x["is_walking"] == True).mean() if "is_walking" in df_x.columns else float("nan")
    alive_only = df_x.filter(pl.col("health") > 0)
    alive_rate = (alive_only["is_walking"] == True).mean() if "is_walking" in alive_only.columns else float("nan")
    print(f"  {label}: all_ticks={all_rate*100:.1f}%  alive_only={alive_rate*100:.1f}%")

# ============================================================
# Q5: zoom_lvl null — weapon breakdown (alive only)
# ============================================================
print("\n" + "=" * 60)
print("Q5: zoom_lvl=null breakdown by weapon_name (alive players)")
print("=" * 60)

alive = df1.filter(pl.col("health") > 0)
zoom_null_by_weapon = (
    alive
    .with_columns(pl.col("zoom_lvl").is_null().alias("zoom_is_null"))
    .group_by(["weapon_name", "zoom_is_null"])
    .agg(pl.len().alias("cnt"))
    .sort("cnt", descending=True)
    .head(30)
)
print("Top weapon+zoom_null combos (alive players):")
print(zoom_null_by_weapon)

# What pct of alive rows are zoom=null?
alive_zoom_null_pct = alive["zoom_lvl"].is_null().mean()
print(f"\nzoom_lvl null rate among alive players: {alive_zoom_null_pct*100:.1f}%")

# ============================================================
# Q6: weapon_name naming check — HE Grenade vs "HE Grenade", C4
# ============================================================
print("\n" + "=" * 60)
print("Q6: Non-standard weapon_name values (vs CS2 standard names)")
print("=" * 60)
STANDARD_NAMES = {
    "AK-47", "M4A1-S", "M4A4", "SG 553", "AUG", "FAMAS", "Galil AR",
    "SSG 08", "AWP", "G3SG1", "SCAR-20",
    "Glock-18", "USP-S", "P2000", "Dual Berettas", "P250", "Five-SeveN",
    "Tec-9", "CZ75-Auto", "Desert Eagle", "R8 Revolver",
    "MAC-10", "MP9", "MP5-SD", "UMP-45", "PP-Bizon", "P90", "MP7",
    "Nova", "XM1014", "Sawed-Off", "MAG-7", "M249", "Negev",
    "Knife", "Bayonet", "Flip Knife", "Gut Knife", "Karambit",
    "M9 Bayonet", "Huntsman Knife", "Falchion Knife", "Bowie Knife",
    "Butterfly Knife", "Shadow Daggers", "Navaja Knife", "Stiletto Knife",
    "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife",
    "Nomad Knife", "Skeleton Knife", "Ursus Knife", "Cord Knife",
    "HE Grenade", "Flashbang", "Smoke Grenade", "Molotov",
    "Incendiary Grenade", "Decoy Grenade", "C4", "Zeus x27", "Defuse Kit",
}

all_weapon_vals = df1["weapon_name"].drop_nulls().unique().to_list()
non_standard = [v for v in all_weapon_vals if v not in STANDARD_NAMES]
print(f"Non-standard weapon_name values in demo1: {non_standard}")
# Check actual names returned
all_vc = df1["weapon_name"].value_counts(sort=True)
print("\nAll weapon_name values (demo1):")
for row in all_vc.iter_rows():
    name, cnt = row[0], row[1]
    ns_flag = " [NON-STANDARD]" if name and name not in STANDARD_NAMES else ""
    print(f"  {str(name):<35} {cnt:>8,}{ns_flag}")

# ============================================================
# Q7: cross-demo consistency — do all demos return same column set?
# ============================================================
print("\n" + "=" * 60)
print("Q7: Column schema consistency across demos")
print("=" * 60)
print(f"Demo1 columns ({len(df1.columns)}): {sorted(df1.columns)}")
print(f"Demo2 columns ({len(df2.columns)}): {sorted(df2.columns)}")
print(f"Demo4 columns ({len(df4.columns)}): {sorted(df4.columns)}")

cols1, cols2, cols4 = set(df1.columns), set(df2.columns), set(df4.columns)
missing_in_2 = cols1 - cols2
missing_in_4 = cols1 - cols4
extra_in_2 = cols2 - cols1
print(f"\nMissing in demo2: {missing_in_2}")
print(f"Extra in demo2: {extra_in_2}")
print(f"Missing in demo4: {missing_in_4}")

print("\nDone.")
