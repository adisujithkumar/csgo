#!/usr/bin/env bash
# hltv_overnight.sh — Scrape HLTV demos in batches, parse immediately, delete RARs.
#
# Workflow per batch:
#   1. Snapshot existing RARs
#   2. Run scraper --max-downloads 10 (downloads new ones to hltv-scrape/data/)
#   3. Detect new RARs via comm (no bash-array bug)
#   4. ETL parse + --delete-source (keeps only ~4 MB Parquet per demo)
#   5. Sleep 30s, repeat
#
# Usage:
#   nohup bash scripts/hltv_overnight.sh >> output/overnight.log 2>&1 &
#   tail -f output/overnight.log
#
# Stop it:  kill $(pgrep -f hltv_overnight)

set -euo pipefail

SCRAPE_DIR=/mnt/d/projects/fps/misc_cs/hltv-scrape
AWP_DIR=/mnt/d/projects/fps/awpy/csgo
LOG=$AWP_DIR/output/overnight.log
MAX_BATCHES=500        # ~5000 demos max; adjust if needed
DEMOS_PER_BATCH=10     # How many fresh demos to scrape per round
BETWEEN_BATCH_SLEEP=30 # seconds between batches (on top of scraper jitter)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p "$AWP_DIR/output"
echo "" >> "$LOG"
log "=== hltv_overnight started (PID $$) ==="
log "SCRAPE_DIR=$SCRAPE_DIR  AWP_DIR=$AWP_DIR"
log "MAX_BATCHES=$MAX_BATCHES  DEMOS_PER_BATCH=$DEMOS_PER_BATCH"

total_downloaded=0
total_parsed=0

for batch in $(seq 1 $MAX_BATCHES); do
    log "--- Batch $batch ---"

    # Snapshot existing RARs with full paths, sorted
    ls "$SCRAPE_DIR/data/"*.rar 2>/dev/null | sort > /tmp/hltv_rar_before.txt || true
    before_count=$(wc -l < /tmp/hltv_rar_before.txt)
    log "RARs on disk before scrape: $before_count"

    # --- Scrape new demos ---
    log "Scraping up to $DEMOS_PER_BATCH demo(s)..."
    cd "$SCRAPE_DIR"
    if ! uv run python mvp_scrape_demos.py \
        --start_url "https://www.hltv.org/results?stars=1" \
        --limit 100 \
        --max-downloads "$DEMOS_PER_BATCH" \
        --timeout 120000 \
        --data-dir data \
        >> "$LOG" 2>&1; then
        log "WARN: scraper exited non-zero (may be done or rate-limited)"
    fi

    # Detect newly downloaded RARs using comm (sorted diff)
    ls "$SCRAPE_DIR/data/"*.rar 2>/dev/null | sort > /tmp/hltv_rar_after.txt || true
    mapfile -t NEW_RARS < <(comm -23 /tmp/hltv_rar_after.txt /tmp/hltv_rar_before.txt)
    after_count=$(wc -l < /tmp/hltv_rar_after.txt)

    log "RARs after scrape: $after_count  |  New: ${#NEW_RARS[@]}"

    if [ ${#NEW_RARS[@]} -eq 0 ]; then
        log "No new RARs — scraper reached end of results or rate-limited. Stopping."
        break
    fi

    for r in "${NEW_RARS[@]}"; do log "  + $r"; done
    total_downloaded=$(( total_downloaded + ${#NEW_RARS[@]} ))

    # --- ETL parse + delete source ---
    log "Running ETL on ${#NEW_RARS[@]} RAR(s) (--delete-source)..."
    cd "$AWP_DIR"
    uv run python scripts/etl_pipeline.py \
        --input "${NEW_RARS[@]}" \
        --output output/full_v2 \
        --workers 2 \
        --delete-source \
        >> "$LOG" 2>&1
    log "ETL done for batch $batch."

    total_parsed=$(( total_parsed + ${#NEW_RARS[@]} ))
    log "Running totals — downloaded: $total_downloaded  parsed+deleted: $total_parsed"

    log "Sleeping ${BETWEEN_BATCH_SLEEP}s before next batch..."
    sleep "$BETWEEN_BATCH_SLEEP"
done

log "=== hltv_overnight finished. Total downloaded: $total_downloaded  parsed: $total_parsed ==="
