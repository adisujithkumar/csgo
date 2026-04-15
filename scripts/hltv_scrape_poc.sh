#!/usr/bin/env bash
# One-shot: scrape 2-3 fresh demos from HLTV results?stars=1, then parse with v2 ETL.
# Logs to output/hltv_poc.log — monitor with: tail -f output/hltv_poc.log

set -euo pipefail

SCRAPE_DIR=/mnt/d/projects/fps/misc_cs/hltv-scrape
AWP_DIR=/mnt/d/projects/fps/awpy/csgo
LOG=$AWP_DIR/output/hltv_poc.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p "$AWP_DIR/output"
echo "" >> "$LOG"
log "=== HLTV PoC scrape+parse started ==="

# Snapshot existing RARs before scraping
cd "$SCRAPE_DIR"
mapfile -t BEFORE < <(ls data/*.rar 2>/dev/null | sort)
log "RARs before scrape: ${#BEFORE[@]}"

# Scrape 2-3 demos (built-in jitter sleep handles rate limiting)
log "Starting HLTV scraper (--limit 3)..."
uv run python mvp_scrape_demos.py \
  --start_url "https://www.hltv.org/results?stars=1" \
  --limit 3 \
  --data-dir data \
  --timeout 90000 \
  >> "$LOG" 2>&1
log "Scraper finished."

# Find newly downloaded RARs by comparing before/after
mapfile -t AFTER < <(ls data/*.rar 2>/dev/null | sort)
NEW_RARS=()
for f in "${AFTER[@]}"; do
  if ! printf '%s\n' "${BEFORE[@]}" | grep -qxF "$f"; then
    NEW_RARS+=("$SCRAPE_DIR/$f")
  fi
done

log "New RARs downloaded: ${#NEW_RARS[@]}"
for r in "${NEW_RARS[@]}"; do log "  $r"; done

if [ ${#NEW_RARS[@]} -eq 0 ]; then
  log "ERROR: No new RARs found. Scraper may have skipped already-downloaded demos."
  log "Trying newest 3 by mtime instead..."
  mapfile -t NEW_RARS < <(ls -t "$SCRAPE_DIR/data/"*.rar 2>/dev/null | head -3 | sed "s|^|$SCRAPE_DIR/|")
  for r in "${NEW_RARS[@]}"; do log "  (fallback) $r"; done
fi

# Parse with v2 ETL
log "Running ETL on ${#NEW_RARS[@]} RAR(s)..."
cd "$AWP_DIR"
uv run python scripts/etl_pipeline.py \
  --input "${NEW_RARS[@]}" \
  --output output/full_v2 \
  --workers 1 \
  >> "$LOG" 2>&1
log "ETL finished."

# Quick validation
log "=== Validation ==="
uv run python - << 'PYEOF' >> "$LOG" 2>&1
import duckdb, pathlib
db = pathlib.Path("output/full_v2/cs2_analytics.duckdb")
if not db.exists():
    print("ERROR: DuckDB not found")
else:
    con = duckdb.connect(str(db), read_only=True)
    print(con.execute("SELECT match_id, source_file, map_name, n_ticks_sampled, schema_version FROM parsed_demos ORDER BY parsed_at DESC LIMIT 5").fetchdf().to_string(index=False))
    print()
    print("weapon_name top-10 (alive):")
    print(con.execute("SELECT weapon_name, COUNT(*) as n FROM ticks WHERE health > 0 GROUP BY 1 ORDER BY 2 DESC LIMIT 10").fetchdf().to_string(index=False))
    con.close()
PYEOF

log "=== DONE ==="
