# Pending Work — Repo Hygiene and Polish

This file tracks repo-hygiene and polish work that is **deferred** at the
time of the initial push to `map-control-awpy2`. It is distinct from
[`./FUTURE_WORK.md`](./FUTURE_WORK.md), which tracks engineering and
research follow-ups for the map-control analytic itself. The items here
are about the shape of the repository / branches / docs, not about new
feature or experiment work.

## Branch structure

The original plan was to ship two stacked branches: `map-control-awpy2`
(implementation + ETL + tests + impl-side docs) and `map-control-research`
(adds analysis scripts + ANALYSIS / FUTURE_WORK / RESEARCH_VISION docs).
We collapsed both into a single `map-control-awpy2` push to move quickly.

**Pending:**

- Split the research-side commits onto a stacked `map-control-research`
  branch. Affected files: `scripts/wp_task*.py`, `scripts/train_wp_model*.py`,
  `scripts/analyze_real_demo.py`, `scripts/quick_test_real_demo.py`,
  `scripts/validate_results.py`, `scripts/reliability_*.py`,
  `scripts/visualize_wp_curves.py`, `scripts/generate_visualizations.py`,
  `scripts/extract_demos.py`, `docs/map_control/ANALYSIS.md`,
  `docs/map_control/FUTURE_WORK.md`, `docs/map_control/RESEARCH_VISION.md`,
  `docs/map_control/analysis/task*.md`, `docs/examples/wp_curves.png`.
- Audit which files belong on which branch in detail (some scripts are
  ambiguous — e.g., `validate_results.py` arguably belongs on impl since
  it validates BFS output, not WP modelling).

## Commit history

The current push contains a single new commit on top of the existing 9
ahead-of-main commits. That single commit is intentionally large to
minimize round-trips during the time-constrained push.

**Pending:**

- Split the squash commit into:
  1. `.gitignore` and the `extract_demos.py` path fix
  2. Tests (`tests/test_map_control_visualization.py`)
  3. Implementation-side docs (`docs/map_control/README.md` + root
     README pointer + impl-side viz artifacts)
  4. Research-side scripts and analysis artifacts (on the new
     `map-control-research` branch once it exists)
  5. Research-side docs
- Consider rewriting the message of the upstream `43eadfb "local chagnes"`
  commit. Low priority — content is good; only the message is rough.

## Documentation polish

The four new docs (`README.md`, `ANALYSIS.md`, `FUTURE_WORK.md`,
`RESEARCH_VISION.md`) were written under time pressure and have not had a
careful editing pass.

**Pending:**

- Read each end-to-end and tighten prose; remove any rough edges.
- Verify all relative file links resolve from their on-branch locations.
  The original plan included an automated link-integrity check; we
  skipped it under time pressure.
- Verify all embedded numerical claims against the source `task*.md`
  files in `docs/map_control/analysis/`.
- Add a short top-level `docs/README.md` index if multiple subdirectories
  accumulate.
- Review code blocks (`python` snippets) for syntactic correctness — the
  original plan included a parse check that we skipped.

## Validation gates skipped during the push

The "on-the-fly validation playbook" agreed in the planning phase
included automated gates at each phase boundary. We skipped most of them
to save time. Specifically:

- **Pre-push test sweep.** We ran `pytest` once at baseline (31/31 passed
  in 52 s) but did not re-run after staging changes. The added test file
  is `tests/test_map_control_visualization.py`; the staged changes
  outside it are docs + scripts (no production code touched), so a
  regression is unlikely but unverified.
- **Hardcoded-path scan in scripts.** We caught one (`extract_demos.py`
  at `/mnt/d/...`) and patched it to use `--data-dir` + `HLTV_DATA_DIR`
  env var. Other scripts may have similar issues we have not audited.
- **Debug-print scan in `awpy/`.** Confirmed once at baseline; not
  re-run after the push.
- **Doc-link integrity** and **redaction term scan** were not run
  programmatically; they were enforced manually during writing.
- **Remote SHA verification.** We will check this after the push;
  recording it here in case it gets skipped.

## Files intentionally not pushed (gitignored)

These are local-only by design — listed here so anyone navigating the
repo understands what is missing relative to a complete reproduction:

- `MAP_CONTROL_FEATURE.md` — early kitchen-sink design doc; superseded
  by the four docs in `docs/map_control/`. Kept locally as scratch.
- `output/` — ETL outputs (parquet, DuckDB) and trained model JSONs.
  ~600 GB of demo data feeds this, regeneratable via
  `scripts/etl_pipeline.py` + `scripts/train_wp_model_v2.py`.
- `test_data/` — extracted real demos.
- `docs/fellows/` — personal application materials.
- `docs/examples/sandbox.ipynb` — exploratory scratch notebook.
- `docs/examples/test.csv`, `docs/examples/tmp/` — scratch artifacts.
- `awpy/data/de_dust2` — stray cross-platform binary that
  `awpy get` deposited; regenerate via `awpy get`.
- `.aca/`, `.claude/`, `tmp/`, `*.log` — local agent / IDE / log files.

## Other small items

- The `save_round_map_control_with_players_gif` docstring at
  `awpy/plot/map_control.py:469` still says "128 tick/sec" which is
  CSGO-era; CS2 demos are 64 tps. Documented in `README.md` under
  Known limitations; should be patched in code.
- `scripts/extract_demos.py` was patched to take `--data-dir` instead of
  hardcoding `/mnt/d/...`. The other scripts (notably the analysis
  ones) still assume relative paths and a particular `output/` layout;
  they have not been audited for portability beyond the original WSL
  workspace.
- The `awpy.visibility.VisibilityChecker` is implemented and tested but
  not wired into BFS — sightline-gated map control is a stub. This is
  flagged in `README.md` (Known limitations) and tracked as Tier 1 work
  in `FUTURE_WORK.md`.

## Suggested order when resuming

1. Verify the push succeeded and the remote tree is what we expect (no
   sensitive files leaked).
2. Run the test suite on the freshly-cloned remote to confirm
   reproducibility.
3. Split the research-side files onto `map-control-research`.
4. Edit-pass on the four docs.
5. Run the validation gates that were skipped.
6. Optionally split the squash commit into the five smaller commits
   listed above.
