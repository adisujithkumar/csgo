# Future Work

A tiered roadmap for the CS2 map control extension to awpy v2. Tier 1 lists
near-term engineering tasks that sharpen the current scalar BFS pipeline. Tier
2 sketches research directions that take the project past scalar reduction
into spatial and structural modeling. The bridge at the end shows how Tier 1
items feed Tier 2.

We are framing this against a preliminary finding logged in the companion
analysis: a polygon-area-weighted scalar map control feature, when added to a
win-probability baseline that already includes `alive_diff`, contributes
~0 AUC. This is a first scalar-level check, not a definitive negative result.
Tier 1 refines the check; Tier 2 questions the scalar reduction itself.

Cross-references: implementation walkthrough in `./README.md`, the analysis
log in `./ANALYSIS.md`, and the deep research expansion of Tier 2 in
`./RESEARCH_VISION.md` (this doc is the index, that doc is the depth).

---

## Tier 1 — Near-term engineering

These are concrete tasks. We have working code for each one to extend; none
of them require new infrastructure. Effort estimates assume one engineer.

### 1. Per-scenario ablation slices

The headline ablation (`scripts/wp_task6_mc_ablation.py`) trains one model
per map across all in-round ticks. Map control might still matter inside
narrower scenarios where alive-count is locally flat — post-plant retakes,
contested mid-round states, eco-vs-full-buy round openings, the last 30
seconds of regulation. We extend the existing ablation harness with a
`--scenario` flag and slice along: `is_bomb_planted == 1`, equip-bucket
pairs (eco/half/full × eco/half/full), kills clustered around a contested
place id, time-decile bins (0-10%, 10-20%, ..., 90-100% of round), and the
late-round window where `(round_end - tick) <= 30s * tickrate`. For each
slice we report paired delta-AUC and delta-Brier with the same group K-fold
seed used in the parent script. Hypothesis: scalar MC adds signal precisely
when alive-count is locally constant and the round depends on positional
tradeoffs. Effort: ~1 engineer-day. Files to touch:
`scripts/wp_task6_mc_ablation.py`, output to
`output/full_v2/analysis/task6_scenarios/`.

### 2. Per-map sensitivity analysis

The current ablation aggregates across maps, but we expect MC's value to
vary with map geometry. Vertical maps (`de_nuke`, `de_vertigo`) compress
many tactical decisions into the Z axis; large open maps (`de_mirage`,
`de_ancient`) spread alive players over more independent regions. We train
per-map models with and without MC, report delta-AUC ordered by map, and
overlay against simple map descriptors (number of nav tiles, mean tile area,
Z spread, place count). Hypothesis: nuke and vertigo show the largest
positive delta because alive-count is a poor proxy for floor control there.
Effort: ~1 engineer-day. Files: extend
`scripts/wp_task6_mc_ablation.py` to dump per-map JSON, add a notebook
under `docs/map_control/analysis/per_map_sensitivity.ipynb`.

### 3. Wire VisibilityChecker into BFS

`awpy/visibility.py` provides `VisibilityChecker.is_visible(start, end)`
backed by a BVH over the map's collision triangles. It is currently
standalone. The `MapControlConfig.sightline_enabled` flag exists but is not
consumed. We wire it through `_conical_bfs`: when enabled, before adding a
neighbor's contribution, call `is_visible(source_centroid, neighbor_centroid)`
and skip if the segment intersects geometry. We cache the per-map
`VisibilityChecker` next to `_nav_cache` to avoid rebuilding the BVH.
Validation uses the existing comparison images at
`docs/examples/map_control_validation/visibility_comparison_*.png` (BFS-only,
visible-only, and triptych) — the new integrated output should match the
"visible-only" reference. Effort: ~2 engineer-days, plus benchmark on a 5s
tick window to confirm we stay under the 10ms-per-frame budget. Files:
`awpy/stats/map_control.py`, `awpy/visibility.py`,
`tests/test_visibility.py`, new
`tests/test_map_control_sightline.py`.

### 4. Smoke / molotov / flash modifiers

Three flags on `MapControlConfig` are currently stubbed:
`smoke_blocking_enabled`, `molotov_denial_enabled`, `flash_weight_enabled`.
The ETL already produces `smokes/<match_id>.parquet` and
`infernos/<match_id>.parquet` (mol/incendiary), each with start/end ticks
and a centroid. Implementation sketch: at frame time, fetch active smokes
and infernos for the tick; in `_conical_bfs` extend the visibility gate so
a tile's contribution is zero when the segment from source centroid to
target centroid passes within `smoke_radius` of an active smoke; tiles
within `molotov_radius` of an active inferno get their per-team
contribution multiplied by zero (full denial). For flashes, scale a
flashed player's whole tile-contribution dictionary by
`flash_weight_multiplier` for the duration the player is blinded; we get
flash duration from the existing `flashes` table. Effort: ~3 engineer-days
(smoke is the cheapest, flash needs duration extrapolation). Files:
`awpy/stats/map_control.py`, new helper `awpy/stats/_utility_state.py`,
extend `tests/test_stats.py`.

### 5. Kill-distance-derived weapon BFS depth

`WEAPON_CONE_PARAMS` currently uses hand-tuned hop depths (AWP 20, rifle 15,
SMG 10, pistol 12, shotgun 5, knife 2). The docstring already cites kill
distances from the `kills` table for each class, but the depth values were
chosen by eye. We compute, per weapon, the median and 75th-percentile
engagement distance from `damages` (more rows than kills) joined to tick
positions, then re-derive depths under a fixed assumption of ~1m per hop on
dust2 (verifiable per-map from the nav). Output is a
`MapControlConfig.cone_params_override` JSON we ship next to the nav files.
We expose a CLI: `python -m awpy.stats.map_control.fit_cone_params --map
de_dust2 --db output/full_v2/cs2_analytics.duckdb`. Effort: ~2
engineer-days. Files: `awpy/stats/map_control.py`,
`scripts/fit_cone_params.py`, output JSON to `awpy/data/cone_params/`.

### 6. Self-improving place mapping

We have a 100% tile-to-place mapping for `de_dust2` — 24 named regions,
generated from 10 demos by aggregating each tile's modal `last_place_name`
across all observed ticks and BFS-filling unlabeled neighbors from labeled
ones. The artifact lives at
`docs/examples/map_control_validation/tile_to_place_dust2_full.json`.
We generalize: a script that takes any map name, ingests N parsed demo
parquets, builds the per-tile vote dictionary, runs the BFS gap-fill, and
writes a JSON keyed by area_id. Coverage is measured as
`1 - count(area_id with no vote and no labeled BFS neighbor) / total_areas`.
We run it on all 9 active competitive maps. Effort: ~2 engineer-days.
Files: new `scripts/build_place_mapping.py`, output to
`awpy/data/place_mappings/<map_name>.json`, mapping dataclass and loader
in `awpy/nav.py`.

### 7. Nav-cache audit and conical-BFS LRU profiling

`map_control.py` keeps three module-level caches (`_nav_cache`,
`_centroid_cache`, `_z_centroid_cache`) plus an `lru_cache(100_000)` on
`_conical_bfs_cached`. We have a `clear_nav_cache()` test fixture, but the
caches are not actively measured. We add: (a) an opt-in
`AWPY_MAP_CONTROL_PROFILE=1` env var that logs cache hit/miss rates at
process exit; (b) a per-process counter on the LRU; (c) a small benchmark
that runs `get_round_map_control_frame_metrics` over one round at the
default 10° yaw bucket and reports `cache_info()`. We expect >95% hit rate
for repeated player tiles within a round. Effort: ~half a day. Files:
`awpy/stats/map_control.py`, new `scripts/profile_map_control.py`.

### 8. Analytics-level test coverage

We have 30 tests in `tests/test_map_control_visualization.py` covering
plotting and overlay output, plus shape-level tests in `tests/test_stats.py`,
but the analytics-layer functions (`get_tick_map_control_tile_values`,
`_bfs`, `_conical_bfs`, `_calculate_map_control_frame_metric`) have no
end-to-end tests against known tile sets. We add a small fixture: three
hand-placed CT players on `de_dust2`, three T players, frozen tick. We
assert: (a) source tiles always have value 1.0; (b) the union of CT-reachable
tiles within `bfs_depth` matches the BFS-by-hand expansion; (c) the
area-weighted frame metric is stable to within float-tolerance across
re-runs; (d) conical BFS with rear halo disabled produces a strict subset of
the value set produced with rear halo enabled. Effort: ~1 engineer-day.
Files: new `tests/test_map_control_analytics.py`,
fixtures under `tests/data/map_control/`.

### 9. Schema-version bump for ETL

`scripts/etl_pipeline.py` writes `schema_version="v2"` and joins parsed
ticks/rounds/kills/damages/bomb/smokes/infernos/grenades/shots into a
DuckDB. v3 adds: `is_post_plant` (per-tick bomb-planted gating), `last_fired_tick`
(per player, ticks since last weapon discharge — useful for utility-vs-weapon
state), `bomb_carrier_steamid`, and a wallbangable-tile cache derived from
`VisibilityChecker`. The bump is ergonomic, not destructive — we keep v2
readers, add v3 columns as additional. Effort: ~2 engineer-days. Files:
`scripts/etl_pipeline.py`, `awpy/data/schemas.py` (new), and a migration
note in `./README.md`.

### 10. Stability of conical-BFS at small yaw buckets

The default `MapControlConfig.yaw_bucket_deg` is 10°, chosen to balance
cache hit rate against accuracy. We have not measured the accuracy cost
empirically. We sweep buckets ∈ {1°, 5°, 10°, 20°} on a fixed test demo
and report: (a) per-frame metric distribution; (b) per-tile metric
disagreement (KL divergence between the bucketed-distribution and the
ungated continuous one); (c) wall-clock cost. The expected outcome is
that 5° preserves nearly all tile-level structure for less than 2x
runtime, and 1° is wasteful. Effort: ~1 engineer-day. Files: new
`scripts/yaw_bucket_sensitivity.py`, output to
`docs/map_control/analysis/yaw_bucket_sensitivity.md`.

### 11. Training-ready DuckDB view

`scripts/train_wp_model_v2.py` rederives features by re-grouping the
ticks parquet at every model run (with a 20-tick stride for a ~5s grid).
This is wasteful. We add a `wp_features_v1` view to the DuckDB layer that
materializes the pre-aggregated per-(match, round, tick) feature row
including `ct_alive`, `t_alive`, `hp_ct`, `hp_t`, `equip_ct`, `equip_t`,
`is_bomb_planted`, plus `ct_map_control` left-joined from
`map_control_ticks.parquet` when present. Downstream training scripts
read the view and choose stride. Effort: ~1 engineer-day. Files:
`scripts/etl_pipeline.py` (view DDL),
`scripts/train_wp_model_v2.py` (use the view).

### 12. Documentation of WP modeling assumptions

`train_wp_model_v2.py` samples every 20th tick within a round (~5s at 64
tick rate) and treats each sample as i.i.d. when fed to XGBoost. This is a
modeling choice, not a fact. We document why: a 5s grid is dense enough to
catch round-state changes (kills, plants) but sparse enough to keep
intra-round autocorrelation manageable, and matches the cadence at which
analysts watch demos. We then test alternatives: 1s, 2s, 5s, and "event-only"
(freeze_end + every kill + plant + defuse + round_end). For each cadence we
report sample count, AUC, and the autocorrelation between consecutive
training rows within a round. Effort: ~1 engineer-day, results into
`docs/map_control/analysis/wp_sampling_cadence.md`.

---

## Tier 2 — Research directions

These take the project past scalar BFS. Each direction is exploratory and
will need iteration, but they have concrete first runs.

### Direction 1 — Multi-channel spatial encoding (no scalar reduction)

The preliminary scalar-MC finding suggests that polygon-area-weighted
averaging collapses the structure that should make MC informative. A
single tile of contested A-site has the same scalar weight as a single tile
of empty CT spawn, even though the former carries far more tactical
information. We propose to abandon scalar reduction at the feature level:
encode the per-tick state as a multi-channel grid where each channel
preserves a distinct part of the structure.

Concrete sketch: rasterize the nav mesh to a fixed 64x64 grid per map
(consistent across runs by storing the grid coordinates next to the nav
JSON). Per cell, emit ~21 channels: per-team alive count touching that
cell (2), per-team economy density (2), per-player BFS-reach depth (10
slots, 5 per team, zeros when fewer alive), weapon-class-conditioned cone
reach max (1 per team), time normalized within round (1), bomb state
encoded as a 3-channel one-hot (planted-A, planted-B, not-planted), and
place id as a categorical embedding (1 channel after embedding). The
result is a (64, 64, 21) tensor per tick. We feed it into a small ConvNet
or a graph attention net that respects nav adjacency. Output is the same
WP target.

Open questions we want to answer: does this beat the alive-diff baseline
on AUC? Does it beat it more in the per-scenario slices from Tier 1 item
1? Is the gain coming from any single channel (we ablate channels one
at a time) or only from interactions? Does a model trained on dust2
transfer to inferno without retraining (probably not), and what does that
tell us about whether the model is learning map-specific tactics or
something more general?

What we run first: train a 5-layer ConvNet on the same match-grouped
K-fold split as the existing baseline; compare AUC on the same eval rows
as `wp_task6_mc_ablation.py`. If the ConvNet under-performs the scalar
baseline, that itself is a strong negative result worth reporting.

### Direction 2 — GNN / attention over hierarchical regions

Map geometry is naturally hierarchical: tiles roll up into places (~24
per map on dust2), places roll up into quadrants and halves. Different
phases of the round care about different levels. Early round, what
matters is which quadrant each side has set up in. Mid round, it's
which place is contested. End round, the tile-level position matters
(who has the off-angle in long doors). A flat scalar collapses all
three levels into one number.

We propose a hierarchical graph attention model. Vertices are tiles,
places, quadrants, halves. Edges follow nav adjacency at each level,
plus parent-child edges across levels. Per vertex, features are the
per-channel multi-channel encoding from Direction 1, aggregated at the
vertex's level (mean for places and above, raw at tiles). At each tick,
we run a multi-head attention pool that learns per-round-phase weights
across levels, then a recurrent update across consecutive ticks of the
same round.

Open questions: how to inject the nav-mesh adjacency into the attention?
Two options — graph attention with explicit edges, or a Transformer with
a binary attention mask derived from the adjacency. Empirically the
latter is easier and often equally accurate for graphs of this size
(~24 places, ~1500 tiles). Does the model attend to higher levels
earlier in the round and lower levels later, and can we visualize
that attention as a tactical readout?

What we run first: build the hierarchical graph with the place mappings
from Tier 1 item 6; train a single-layer attention pool with no
recurrence, just to confirm the graph structure is well-formed and the
WP target is learnable. Then layer in temporal updates.

### Direction 3 — Win probability + SHAP for tactical insight

Even if scalar MC is not a top global feature, it might dominate in
specific tactical states. Consider a 1v3 post-plant retake: the alive
diff is fixed at -2 for the whole window, but the lone defender's map
control changes dramatically as they reposition. A globally-weak feature
can be locally dominant. We propose using SHAP attribution on a model
trained over state clusters rather than over all rows.

Sketch: cluster per-tick states using k-means on a low-dimensional
embedding (UMAP of the 21-channel spatial encoding from Direction 1, or
a simpler hand-crafted feature set: alive_diff, equip_diff, time_pct,
is_bomb_planted, ct_map_control). Train one mini-model per cluster.
Compute SHAP values for each mini-model. Rank features by their cluster-
mean SHAP magnitude. Emit a heatmap: rows = clusters, columns = features,
cell value = SHAP magnitude.

Open questions: do post-plant retake clusters show MC dominance? Do
opening-pick clusters show the opposite (alive_diff dominates)? Are the
clusters interpretable enough for an analyst to label them
("lurk-and-flank", "fast-A execute", "default-with-info")? Can the
heatmap be turned into a deployable pre-round briefing tool?

What we run first: K-means on the existing baseline features with K
chosen to maximize silhouette; SHAP on each cluster's mini-model;
manual labeling of the top 10 clusters by inspection of representative
ticks. This is feasible within ~1 week and is the most readable first
output of this whole research direction.

### Direction 4 — IGL navigation model

Once we have a stable WP estimator, we use it as a reward signal to train
a policy that issues sub-tick spatial commands — an "in-game leader"
model. The architecture is value-then-policy: the WP model is the value
function; an LSTM over per-tick state outputs a target tile distribution
per CT player at each freeze_end. We train the policy first via
behavioral cloning on pro demos (target = the actual tile the player was
on 30 ticks later), then fine-tune via a policy-gradient objective
where the reward is the change in expected WP at round end.

Sketch: state is the multi-channel encoding from Direction 1; policy
output is a softmax over tile ids per CT player; loss is cross-entropy
to the actual future-tile during BC, then advantage-weighted log-prob
during fine-tune. We restrict to freeze_end-time decisions to keep the
problem tractable; sub-tick decisions during a round are out of scope
for v0.

Open questions: does the policy converge to recognizable strategies? On
dust2, does CT default cluster into "stack-A", "stack-B", "two-and-two
with rotator", "double-AWP-mid"? On inferno, does it find the
"banana-stack" and "apartments-pinch" strategies that are part of the
pro meta? Is the BC loss alone enough, or does the fine-tune phase
actually move the policy in a measurable direction? What's the WP gain
of the trained policy vs. a uniform-random baseline?

What we run first: pure BC on dust2 freeze_end states from the existing
ETL output; visualize the predicted tile distribution per CT player on
a sample of 20 rounds; confirm the predictions are not collapsing to a
single tile. Fine-tune is the second milestone.

### Direction 5 — Strategy clustering across pros, leagues, patches

The output of Direction 4 is a policy. The output of strategy clustering
is a typology — a label per round indicating which strategic family it
belongs to. We can build the typology either from the IGL policy
(softmax outputs at freeze_end clustered across rounds) or from raw demo
trajectories (per-player place trajectories aligned by dynamic time
warping, then clustered).

Sketch: per round, build a per-player place sequence at 1s resolution
from freeze_end to bomb-plant or round end; align across players using
DTW; cluster the resulting per-round trajectory tensors with HDBSCAN.
Track cluster prevalence per team, per league, per patch over time;
detect meta-shifts as discontinuities in cluster prevalence.

Open questions: are clusters stable across teams (one universal
"slow-default") or team-specific (each top-10 team has signature
strategies)? Can we forecast meta-shifts by detecting clusters that
spike in a few teams before the rest of the league adopts them? Does a
patch that nerfs M4A1-S correlate with a measurable cluster shift on
CT side at long-engagement maps?

What we run first: round-level clustering on the HLTV BO3 corpus,
restricted to dust2; manual labeling of the top 15 clusters; year-over-
year prevalence plots per top-10 team. This is the most "analyst-facing"
direction — the visualizations are the deliverable, the model is just
the substrate.

---

## Bridge — How the tiers compose

Tier 1 items are not throwaway hardening. Each one is a building block for
something in Tier 2.

```
                      Tier 1                                 Tier 2
                    ----------                              -------
       1. per-scenario slices  ---------> 3. SHAP per cluster
       2. per-map sensitivity  ---------> 1. multi-channel encoding
                                          (per-map ablation evidence)
       3. sightline BFS        ---------> 1. multi-channel encoding
                                          (visibility-aware reach channel)
       4. utility modifiers    ---------> 1. multi-channel encoding
                                          (smoke/molly/flash channels)
       5. weapon depth fit     ---------> 1. multi-channel encoding
                                          (cone-reach channel calibration)
       6. place mappings       ---------> 2. hierarchical GNN
                                          (place tokens for attention)
       7. cache audit          ---------> 1, 2, 4 (training throughput)
       8. analytics tests      ---------> all (regression safety)
       9. schema v3            ---------> 4. IGL nav model
                                          (post-plant flag + last-fired)
      10. yaw bucket           ---------> 1, 2 (cone-reach channel)
      11. training-ready view  ---------> 3, 4 (faster eval cycles)
      12. sampling cadence     ---------> 4. IGL nav model
                                          (action time-step choice)
```

Read top-down: each Tier 2 direction has Tier 1 prerequisites. The
multi-channel encoding (2.1) needs sightline-gated BFS (1.3), utility
modifiers (1.4), data-driven cone depths (1.5), and the yaw-bucket
sensitivity sweep (1.10) to know its features are calibrated. The
hierarchical GNN (2.2) cannot exist without place mappings on every map
(1.6). The SHAP-per-cluster work (2.3) is most informative when paired
with the per-scenario slices (1.1) — the slices tell us which clusters
already show MC dominance under the scalar model, which in turn guides
where we expect SHAP to find structure.

The IGL navigation model (2.4) and strategy clustering (2.5) sit further
out. They need the schema v3 columns (1.9), the training-ready view (1.11),
and a clear position on sampling cadence (1.12). They also rely on the
WP model from Direction 3 being well-calibrated, so we sequence them
after 2.1 and 2.3 produce a stable value function.

We expect Tier 1 items 1, 2, 7, 8 to land in the next development cycle.
Items 3, 4, 6, 11 are second-cycle. Items 5, 9, 10, 12 are third.
Tier 2 directions begin once Tier 1's first cycle is in main and the
corresponding prerequisites are met.
