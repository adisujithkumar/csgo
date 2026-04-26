# Map Control — Preliminary Research Log

> **Status: preliminary investigation log.** This document records what we
> have actually run on a CS2 map-control implementation and what we believe
> the data is telling us so far. It is **not** a finalized study. The
> analyses described here operate at scalar / round-level abstractions; more
> nuanced per-scenario, per-map, and contextual investigations are pending
> and tracked in [`./FUTURE_WORK.md`](./FUTURE_WORK.md).

## 1. Overview and caveats

We ported a BFS-based map-control feature from the legacy CS:GO `awpy`
codebase to the current CS2 `awpy` v2 stack (Polars, `demoparser2`,
Python 3.11+) and built an ETL + DuckDB analytics layer on top of it. With
that infrastructure in place we ran a first batch of empirical questions
against a corpus of **1,697 demos across 9 competitive CS2 maps** drawn
from the HLTV professional scrape:

| map | demos |
| --- | --- |
| de_mirage | 279 |
| de_ancient | 277 |
| de_dust2 | 271 |
| de_nuke | 268 |
| de_inferno | 260 |
| de_anubis | 206 |
| de_train | 104 |
| de_vertigo | 33 |
| de_overpass | 14 |

The corpus was parsed end-to-end with our v2 schema (29-column tick table,
plus per-match parquet for `rounds`, `kills`, `damages`, `bomb`, `smokes`,
`infernos`, `grenades`, `shots`, `player_economy`, `flashes`,
`he_grenades`).

The central empirical question we set out to answer was:

> **Does the BFS map-control scalar add predictive signal beyond simpler
> features (alive count, HP, equipment, time, plant state) for round
> outcome?**

The honest preliminary answer, at the scalar / per-tick level we tested:
**not meaningfully, once `alive_diff` is already in the model.** Adding
`ct_map_control` as a single extra XGBoost feature on top of a
ten-feature baseline produced ΔAUC values ranging from **-0.0056 to
+0.0114** across all nine maps in our paired-fold ablation. The headline
table is in §7.

We want to be careful about how this finding is read. It is not a claim
that map control "doesn't matter." It is a claim that **a single
polygon-area-weighted scalar of CT-controlled space is largely redundant
with `alive_diff` once `alive_diff` is present in the model.** The right
reading is:

- The current scalar formulation collapses spatial structure into one
  number; that compression may itself be the issue.
- We have not yet investigated slices where map control plausibly carries
  unique information (post-plant rounds, eco rounds, contested chokepoints,
  specific maps and sites).
- An unscaled / multi-channel / structural representation of the same BFS
  output may behave very differently when given to a model that can attend
  to it.

These are the directions we plan to explore next, and they motivate the
roadmap in [`./FUTURE_WORK.md`](./FUTURE_WORK.md).

**Reading guide.** §2 documents the data infrastructure the rest of the
analyses ride on. §3 covers the per-map win-probability models that
establish a baseline. §4–§6 describe the three positive-signal
investigations we ran (kill context, place value, swing analysis). §7 is
the negative-result ablation. §8 covers the implementation-correctness
validation that is distinct from the WP modelling. §9 is a short bridge to
[`./FUTURE_WORK.md`](./FUTURE_WORK.md).

For a description of the BFS implementation itself see
[`./README.md`](./README.md). For the long-form research direction see
[`./RESEARCH_VISION.md`](./RESEARCH_VISION.md).

## 2. Data infrastructure

The ETL pipeline lives at
[`scripts/etl_pipeline.py`](../../scripts/etl_pipeline.py) and operates as
a multiprocess worker pool over `.dem` and `.rar` inputs. Each match is
parsed once with `demoparser2` (via `awpy.Demo`) and written to per-match
parquet files. The outputs are then registered as DuckDB views via
`SELECT ... FROM read_parquet('output/full_v2/<table>/*.parquet',
union_by_name=true)`. We chose per-match parquet over a monolithic file
to avoid the O(n²) rewrite cost that bites monolithic parquet pipelines as
the corpus grows; this design was validated on the initial 10-demo
smoke run before scaling to the full 1,697-demo corpus.

**Tick schema (29 columns, v2).** Each tick is sampled every 16 game ticks
(four samples per second at 64-tick) and persisted with:

- Identity: `tick`, `steamid`, `name`, `round_num`, `side`
- Position: `X`, `Y`, `Z`, `place`
- Health / armor: `health`, `armor`
- Economy: `current_equip_value`, `has_defuser`, `has_helmet`
- Active weapon: `weapon_name`
- Movement: `velocity_X`, `velocity_Y`, `velocity_Z`, `yaw`, `pitch`
- Player state: `flash_duration`, `zoom_lvl`, `is_walking`, `duck_amount`,
  `spotted`
- World state (broadcast to every player row at the same tick):
  `is_bomb_planted`, `which_bomb_zone`, `game_time`

Beyond ticks we maintain per-match parquet for `rounds`, `kills`,
`damages`, `bomb`, `smokes`, `infernos`, `grenades`, `shots`,
`player_economy`, `flashes`, and `he_grenades`.

**Reliability audit.** Before trusting any field we ran a two-stage
reliability check, partly because the v2 schema introduced fields that
were not exercised in the legacy pipeline. The first stage
([`scripts/reliability_analysis.py`](../../scripts/reliability_analysis.py))
re-parses 12 demos with the full prop list, checks coverage and null
rates, and verifies that columns we expect are actually present. The
second stage
([`scripts/reliability_followup.py`](../../scripts/reliability_followup.py))
drills into the surprising patterns surfaced by the first.

The follow-up findings that matter for downstream analysis:

- **`weapon_name` null rows.** The non-trivial null fraction in
  `weapon_name` is dominated by dead-player rows (health == 0). This is a
  legitimate artifact of how `demoparser2` exposes the field — a player
  with no active weapon entity has nothing to report. Filtering on
  `health > 0` recovers a clean, near-100%-coverage column. Downstream
  consumers (the conical-BFS weapon class router, the kill-context
  analysis) all condition on alive players for unrelated reasons, so this
  null pattern is consistent with their assumptions.
- **`zoom_lvl` interactions.** Non-zero zoom on rifles is a legitimate
  signal only for SG553 and AUG (which scope), not for the broader rifle
  pool. Treating zoom as a generic signal across all primaries would
  double-count an attribute that is structural to two specific weapons; we
  noted this and use weapon-class-conditional logic where zoom matters.
- **`spotted` on dead players.** The follow-up confirms that the few
  dead-spotted == True rows are a death-tick edge: the spotted flag is set
  on the tick when the player dies, before the entity is fully torn down.
  We treat this as artefactual.
- **Schema consistency.** Cross-demo checks across three reference demos
  confirm column-set parity, so DuckDB's `union_by_name` glob is safe; we
  do not have to special-case missing columns at query time.

These checks are not glamorous but they save us from reading mechanical
artifacts as analytic findings later, especially when we start slicing
fine.

## 3. Win probability modelling

The training entry point is
[`scripts/train_wp_model_v2.py`](../../scripts/train_wp_model_v2.py).
We train **one XGBoost classifier per map** rather than a single
cross-map model: per-map structure (chokepoint geography, default
strategies, bombsite shapes) varies enough that we expect the model to
exploit features differently per map, and we want the per-map AUCs as a
diagnostic.

**Setup.**

- **Target:** `ct_won` per (match, round, tick) row. Constructed from the
  `rounds` table's `winner` field equal to `'ct'`.
- **Features:**
  `round_time_pct`, `ct_alive`, `t_alive`, `hp_ct`, `hp_t`, `equip_ct`,
  `equip_t`, `alive_diff` (= `ct_alive - t_alive`), `equip_diff`,
  `is_bomb_planted`, `ct_map_control` (filled with `0.5` when missing in
  the baseline runs since BFS coverage is not 100%).
- **Sampling:** ticks are downsampled within each round at stride 20 from
  the 16-tick parquet sample (so roughly one row every 5 seconds of game
  time) to avoid the model fitting tick-level autocorrelation.
- **Cross-validation:** 5-fold `GroupKFold` keyed on `match_id`. This is
  the right grouping: leakage between train/test on the same match would
  be massive given that adjacent ticks within a round are highly
  correlated. The same fold seed is reused for the ablation (§7) so the
  paired comparisons are honest.
- **XGBoost hyperparameters:** `n_estimators=150`, `max_depth=5`,
  `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`,
  `eval_metric=logloss`, `tree_method=hist`. Not tuned per map; we wanted
  a defensible default rather than a per-map race to the top.

**Per-map results.** The numbers below are mean AUC and mean Brier across
the five folds. Top-feature is the largest XGBoost gain importance per
map.

| map | matches | rounds | ticks | mean_auc | mean_brier | top_feature |
| --- | --- | --- | --- | --- | --- | --- |
| de_mirage | 263 | 5,512 | 98,522 | 0.8666 | 0.1482 | alive_diff (0.43) |
| de_ancient | 262 | 5,438 | 92,582 | 0.8659 | 0.1484 | alive_diff (0.46) |
| de_dust2 | 257 | 5,218 | 97,695 | 0.8574 | 0.1523 | alive_diff (0.48) |
| de_inferno | 251 | 5,106 | 100,693 | 0.8598 | 0.1507 | alive_diff (0.51) |
| de_nuke | 250 | 5,027 | 90,253 | 0.8555 | 0.1541 | alive_diff (0.50) |
| de_anubis | 201 | 4,067 | 77,291 | 0.8481 | 0.1539 | alive_diff (0.49) |
| de_train | 94 | 2,004 | 35,318 | 0.8436 | 0.1611 | alive_diff (0.46) |
| de_vertigo | 33 | 730 | 13,217 | 0.8231 | 0.1754 | alive_diff (0.36) |

`de_overpass` is excluded from this table because it falls below the 30-demo
threshold the training script enforces (we have only 14 demos for it in the
current corpus).

The figure at [`../examples/wp_curves.png`](../examples/wp_curves.png)
shows representative per-round win-probability curves across baseline and
+map_control models for a handful of rounds, and visualizes how little
effect the map-control feature has on the curve shape once the other
features are present.

> *Figure 1: Win probability comparison across baseline and +map_control
> models on representative rounds. The two curves track each other closely
> across the round, with the largest divergences occurring at moments
> where a kill simultaneously reshapes spatial control and the alive
> count.*

**The dominant-feature finding.** Across every single map, `alive_diff`
is the highest-gain feature, typically by a factor of two to three over
the next contender (`equip_diff`). In two maps it carries more than half
the total importance (`de_inferno` 0.51, `de_nuke` 0.50). `is_bomb_planted`
is consistently the third most important. `ct_map_control` registers gain
0.0 on every map except `de_dust2`, where it gets a token 0.0078 — and
even there `alive_diff` carries 0.48.

This is consistent with a simple structural fact: at the scalar level,
**`alive_diff` is the most legible summary of "who has the round," and
`ct_map_control` as a single number is highly correlated with it in
expectation.** When two players die from the trailing team without
trading, the BFS scalar moves in the same direction as `alive_diff` — and
the model has already eaten the `alive_diff` signal, so the marginal
information of the BFS scalar is small. The interesting question is
whether map control carries signal *orthogonal* to `alive_diff` in
specific scenarios; that is what §7's preliminary ablation cannot answer
and what the slice-level work in [`./FUTURE_WORK.md`](./FUTURE_WORK.md) is
designed to.

The trained per-map model JSON files (with feature importances and
per-fold AUC / Brier / log-loss) live at
`output/full_v2/analysis/wp_models/{map}.json`. They are not committed —
they are regeneratable from the raw parquet by re-running the training
script — but the per-map metrics summary at
[`./analysis/task3_wp_summary.md`](./analysis/task3_wp_summary.md)
captures the headline numbers.

## 4. Kill-context counterfactuals (strongest positive signal)

The strongest positive signal we have surfaced so far comes from a
classification of every kill into three buckets based on the alive-count
state *immediately before* the kill resolves. The pipeline is in
[`scripts/wp_task1_kill_context.py`](../../scripts/wp_task1_kill_context.py)
and the full table is at
[`./analysis/task1_kill_context.md`](./analysis/task1_kill_context.md).

**Method.** We reconstruct, from the per-round `kills` table, the
state of (`ct_alive_before_kill`, `t_alive_before_kill`) at every kill
event using a window function partitioned by `(match_id, round_num)` and
ordered by `(tick, kill_index)`. From that we derive, for each kill from
the attacker's POV:

- `advantage` — the attacker's team has more alive players than the
  victim's team before this kill resolves.
- `parity` — equal alive counts.
- `disadvantage` — attacker's team is down before the kill.

We then ask: conditional on a kill happening at this state, **what is
the round win rate of the attacker's team?**

Across **122,704 non-team kills** in the corpus:

| kill_context | kills | attacker_wr | avg_time_pct |
| --- | --- | --- | --- |
| advantage | 44,707 | **0.958** | 0.777 |
| parity | 39,974 | **0.746** | 0.570 |
| disadvantage | 36,534 | **0.309** | 0.671 |

Broken down by attacker side:

| attacker_side | kill_context | kills | attacker_wr |
| --- | --- | --- | --- |
| ct | advantage | 27,005 | 0.949 |
| ct | parity | 21,239 | 0.701 |
| ct | disadvantage | 15,836 | 0.258 |
| t | advantage | 17,702 | 0.973 |
| t | parity | 18,735 | 0.797 |
| t | disadvantage | 20,698 | 0.348 |

A few things make this signal interesting beyond the headline numbers.

**(a) Magnitude.** A 65-percentage-point spread between advantage and
disadvantage attacker WR (0.958 vs 0.309) is a much larger effect than
anything the BFS scalar contributes in §7. Even within parity, the
T-side advantage is striking: T-attackers in parity win 0.797 of rounds
versus 0.701 for CT-attackers, suggesting parity is asymmetric and
favours the T side at the moment of kill resolution. This may reflect
that T parity often follows successful executes, whereas CT parity
follows successful holds; the structural meanings are different.

**(b) Counterfactual asymmetry.** The disadvantage win rate (0.309) is
not 0 — it is meaningfully above zero. This implies that
disadvantage-state kills carry recoverable rounds, and presumably the
shape of the recovery is exactly the kind of thing a richer map-control
representation could illuminate (a 1v3 retake from a position with
strong structural control should differ in win rate from a 1v3 push out
of T main with no map presence). We have not sliced this yet; it is one
of the things [`./FUTURE_WORK.md`](./FUTURE_WORK.md) tracks.

**(c) Why this matters for the negative result in §7.** The kill-context
signal demonstrates that round-outcome variance has clearly identifiable
structure, and that structure is recoverable from sufficiently
expressive features. It is therefore not the case that round outcome is
"already saturated" by `alive_diff` and that no further signal is
available. The narrower claim is that the *specific* polygon-area BFS
scalar we evaluated does not surface that additional signal once
`alive_diff` is in the model. The implication is that the form of the
map-control feature, not the existence of map control as a phenomenon,
is what is at issue.

The per-kill table (one row per kill with all derived columns: side,
context, place, time pct, advantage delta, round outcome, etc.) is
materialized at
`output/full_v2/analysis/kills_with_context.parquet` and feeds §5 and
§6. It is not committed — same regeneratability story as the model files.

## 5. Place-value proxy

Before we built the real BFS pipeline (§6) we wanted a scalar position
proxy that did not depend on map-control geometry, partly to sanity-check
the kill-context signal and partly to ask "when in a round does
positional state matter most?" The pipeline is in
[`scripts/wp_task2_time_importance.py`](../../scripts/wp_task2_time_importance.py)
and the full table is at
[`./analysis/task2_time_importance.md`](./analysis/task2_time_importance.md).

**Method.** We sample tick-level rows uniformly across rounds, bucket
into deciles of `round_time_pct`, and compute per-decile correlations
between four candidate features and `ct_won`:
`alive_diff`, `hp_diff`, `equip_diff`, `is_bomb_planted`. We also fit a
per-decile univariate logistic regression on `alive_diff` to give a
clean readout of "by how much does `alive_diff` shift the log-odds of
`ct_won` as the round progresses."

**Result.** Correlations rise smoothly and monotonically with time:

| decile | n_ticks | corr_alive_diff | corr_hp_diff | corr_equip_diff | corr_is_bomb_planted |
| --- | --- | --- | --- | --- | --- |
| 0 | 20,857 | 0.029 | 0.031 | 0.394 | nan |
| 1 | 18,031 | 0.142 | 0.137 | 0.399 | nan |
| 2 | 17,957 | 0.245 | 0.250 | 0.433 | nan |
| 3 | 18,028 | 0.335 | 0.342 | 0.467 | -0.030 |
| 4 | 17,892 | 0.405 | 0.413 | 0.506 | -0.095 |
| 5 | 17,610 | 0.498 | 0.494 | 0.565 | -0.176 |
| 6 | 17,916 | 0.589 | 0.579 | 0.619 | -0.318 |
| 7 | 17,949 | 0.661 | 0.651 | 0.674 | -0.482 |
| 8 | 17,807 | 0.735 | 0.724 | 0.731 | -0.500 |
| 9 | 17,880 | 0.813 | 0.793 | 0.791 | -0.550 |

`corr_is_bomb_planted` is `nan` in early deciles because the bomb is not
plantable yet there. Where it is defined it is *negative* and grows in
magnitude as the round progresses — once a bomb is on the ground,
`ct_won` is increasingly disfavoured. In the final decile the correlation
reaches -0.55. This is the expected sign (post-plant rounds favour T) but
the magnitude is informative: in the late round, plant state alone
flips the conditional probability strongly.

The per-decile logistic regression on `alive_diff`:

| decile | coef_alive_diff | intercept | brier_rmse |
| --- | --- | --- | --- |
| 0 | 0.627 | -0.132 | 0.499 |
| 1 | 0.739 | -0.151 | 0.494 |
| 2 | 0.829 | -0.169 | 0.484 |
| 3 | 0.947 | -0.172 | 0.470 |
| 4 | 1.012 | -0.158 | 0.456 |
| 5 | 1.187 | -0.160 | 0.430 |
| 6 | 1.360 | -0.074 | 0.396 |
| 7 | 1.556 | -0.049 | 0.362 |
| 8 | 1.844 | -0.205 | 0.315 |
| 9 | 2.403 | -0.792 | 0.241 |

The coefficient on `alive_diff` rises by a factor of ~3.8 from decile 0 to
decile 9: each additional alive teammate is worth ~0.63 logits early-round
and ~2.40 logits late-round. The intercept also shifts negative, which
captures the post-plant T-favour effect. Brier RMSE drops from 0.499
(near-random) to 0.241 (informative) over the same span. This is the
clean "when does state matter most" picture: increasingly, with the
biggest gains in the last three deciles.

**Top victim_place.** As a positional proxy, we tabulate `ct_win_rate` and
`t_win_rate` conditional on a kill happening at each `victim_place`
(filtered to `n_kills >= 30`). The top CT-favoured kill places (where a
CT kill at that location predicts a CT round win):

`de_overpass` Construction (0.889), `de_nuke` Hell (0.841), `de_overpass`
LowerPark (0.835), `de_overpass` Canal (0.816), `de_nuke` Vents (0.810),
`de_nuke` Control (0.809), `de_mirage` Apartments (0.801), `de_nuke` Hut
(0.801).

The top T-favoured kill places: `de_anubis` CTSpawn (0.907), `de_anubis`
CTSideUpper (0.886), `de_anubis` Street (0.850), `de_nuke` Vending
(0.827), `de_mirage` Shop (0.826), `de_anubis` Alley (0.821).

**Disadvantage swing per place.** A small subtable in
[`./analysis/task2_time_importance.md`](./analysis/task2_time_importance.md)
computes, for each (map, victim_place), the difference
`wr_disadvantage - wr_parity`. The largest negative swings (the places
where being the down team and getting a kill there *still* doesn't help)
include `de_dust2` Hole (-0.330), `de_anubis` Connector (-0.327),
`de_nuke` Heaven (-0.321), `de_ancient` SideEntrance (-0.316), `de_mirage`
Connector (-0.316), `de_nuke` Observation (-0.311). These are the places
where a "good kill" doesn't actually salvage the round — i.e. places
where the structural problem persists despite the trade. They are
candidates for places where richer map control is most likely to add
information.

Materialized at
`output/full_v2/analysis/place_value_proxy.parquet`.

## 6. Map-control swing analysis

This is the analysis that paired the *real BFS output* with kill events,
to ask: when an `advantage`-state kill happens, does the BFS scalar move
in the way you would expect? The pipeline is in
[`scripts/wp_task5_swing_analysis.py`](../../scripts/wp_task5_swing_analysis.py)
and the full table at
[`./analysis/task5_swing_analysis.md`](./analysis/task5_swing_analysis.md).

**Precompute.** Computing BFS at every tick is too expensive
(~5 ms/call); we precompute it only at the moments that matter.
[`scripts/wp_task4_bfs_precompute.py`](../../scripts/wp_task4_bfs_precompute.py)
extracts target ticks from the kills + rounds tables — every kill tick,
every `freeze_end` tick, every `round_end` tick — and runs the conical
BFS once per target. The cache in `_conical_bfs_cached` re-uses computed
source-tile expansions across ticks, which is effective because hold-angle
players spend many ticks on the same source tile. The output is a
parquet with one row per (match_id, round_num, target_tick, kind) and
the columns `ct_map_control`, `n_ct_tiles`, `n_t_tiles`, etc. for
diagnostics. We keep both a full version and a 5-match-per-map sample
(the sample is what the ablation in §7 uses for tractable training
times).

**Pairing.** For each kill we look up the BFS scalar at the immediately
prior target tick (within the same round) and the BFS scalar at the kill
tick itself. The difference is `ct_mc_delta` (or, from the attacker's POV
when `attacker_side == 'ct'`, the attacker swing equals
`ct_mc_delta`; when `attacker_side == 't'` it is `-ct_mc_delta`).

**Results (6,350 paired kills).**

| attacker_side | kill_context | n | mean_swing | median_swing | std_swing | attacker_wr |
| --- | --- | --- | --- | --- | --- | --- |
| ct | advantage | 1,309 | +0.079 | +0.051 | 0.156 | 0.952 |
| ct | parity | 1,132 | +0.071 | +0.036 | 0.192 | 0.693 |
| ct | disadvantage | 878 | -0.018 | -0.010 | 0.144 | 0.233 |
| t | advantage | 978 | +0.111 | +0.090 | 0.164 | 0.973 |
| t | parity | 991 | +0.014 | +0.014 | 0.193 | 0.802 |
| t | disadvantage | 1,062 | +0.016 | +0.014 | 0.132 | 0.363 |

The directional pattern is correct: kills made from advantage or parity
swing CT control upward when CT is the attacker, and the largest mean
swings are on `advantage` kills (T-attacker advantage kills swing 0.111;
CT-attacker advantage kills swing 0.079). Disadvantage kills have
near-zero or slightly negative mean swing — consistent with the reading
that "a single trade kill while down doesn't reorganize the
contested-territory geometry."

**Correlation between swing magnitude and round outcome.**

| kill_context | n | corr_swing_round_win | mean_swing | win_rate |
| --- | --- | --- | --- | --- |
| disadvantage | 1,940 | 0.013 | +0.001 | 0.304 |
| parity | 2,123 | 0.034 | +0.045 | 0.744 |
| advantage | 2,287 | 0.054 | +0.093 | 0.961 |

Honestly: these correlations are weak. Within any kill-context bucket,
the magnitude of the BFS swing is a noisy predictor of whether the
attacker's team wins the round. The strongest correlation we see is
0.054 (advantage bucket).

The natural interpretation: the alive-count delta carries most of the
"kill-state information," and the BFS swing layered on top mostly
correlates with that same alive-count change rather than carrying
something new. This is the first place we saw the redundancy that §7
formalizes.

**Honest framing.** Swing magnitude alone, conditioned on kill context,
does not predict round outcome strongly. That is one signal at one level
of abstraction (a single scalar swing across one kill event). It does not
tell us whether the *spatial pattern* of the swing — which regions
flipped, whether the swing creates a sightline, whether it splits the
T-side approach paths — carries unique information. That is in
[`./FUTURE_WORK.md`](./FUTURE_WORK.md).

## 7. MC ablation — preliminary negative check

This is the experiment whose result motivates much of
[`./FUTURE_WORK.md`](./FUTURE_WORK.md). The pipeline is in
[`scripts/wp_task6_mc_ablation.py`](../../scripts/wp_task6_mc_ablation.py)
and the full table at
[`./analysis/task6_mc_ablation.md`](./analysis/task6_mc_ablation.md).

**Method.** We restrict the training data to the subset of matches with
BFS coverage (the precomputed parquet from §6's `wp_task4` run), and per
map we train two models in a paired fashion:

- **Baseline.** Ten features:
  `round_time_pct`, `ct_alive`, `t_alive`, `hp_ct`, `hp_t`, `equip_ct`,
  `equip_t`, `alive_diff`, `equip_diff`, `is_bomb_planted`.
- **+MC.** Eleven features: baseline plus `ct_map_control`.

Both models use the same XGBoost hyperparameters as §3 and the same
match-keyed `GroupKFold` split (same seed) so the comparison is paired.
We report mean AUC and mean Brier across folds, and the deltas
(`d_auc = mc_auc - base_auc`, `d_brier = mc_brier - base_brier`).

We run two modes per map: `all_rows` (every tick row, with `ct_map_control`
filled to `0.5` where coverage is missing) and `mc_only` (rows with real
BFS coverage only; coverage = 100% by construction).

**Results.**

| map | mode | n_rows | mc_cov | base_auc | +mc_auc | d_auc | base_brier | +mc_brier | d_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de_inferno | all_rows | 51,023 | 12.8% | 0.8460 | 0.8450 | -0.0009 | 0.1656 | 0.1673 | +0.0018 |
| de_inferno | mc_only | 6,539 | 100.0% | 0.8704 | 0.8700 | -0.0004 | 0.1499 | 0.1485 | -0.0014 |
| de_ancient | all_rows | 30,107 | 16.6% | 0.8165 | 0.8119 | -0.0046 | 0.2087 | 0.2062 | -0.0025 |
| de_ancient | mc_only | 5,007 | 100.0% | 0.8576 | 0.8574 | -0.0002 | 0.1812 | 0.1827 | +0.0015 |
| de_overpass | all_rows | 47,675 | 12.3% | 0.7677 | 0.7622 | -0.0054 | 0.2269 | 0.2259 | -0.0010 |
| de_overpass | mc_only | 5,865 | 100.0% | 0.8306 | 0.8305 | -0.0001 | 0.1794 | 0.1751 | -0.0042 |
| de_anubis | all_rows | 42,429 | 14.0% | 0.7890 | 0.7881 | -0.0008 | 0.2101 | 0.2116 | +0.0015 |
| de_anubis | mc_only | 5,933 | 100.0% | 0.8340 | 0.8284 | -0.0056 | 0.1772 | 0.1807 | +0.0034 |
| de_dust2 | all_rows | 36,749 | 13.6% | 0.8245 | 0.8238 | -0.0007 | 0.2059 | 0.2049 | -0.0010 |
| de_dust2 | mc_only | 4,996 | 100.0% | 0.8530 | 0.8644 | **+0.0114** | 0.1840 | 0.1840 | -0.0000 |
| de_mirage | all_rows | 31,600 | 15.4% | 0.8680 | 0.8667 | -0.0013 | 0.1617 | 0.1616 | -0.0001 |
| de_mirage | mc_only | 4,864 | 100.0% | 0.8951 | 0.8940 | -0.0011 | 0.1383 | 0.1387 | +0.0004 |
| de_nuke | all_rows | 38,340 | 15.0% | 0.7993 | 0.8026 | +0.0033 | 0.2180 | 0.2155 | -0.0025 |
| de_nuke | mc_only | 5,753 | 100.0% | 0.8214 | 0.8269 | +0.0056 | 0.1986 | 0.1933 | -0.0054 |
| de_train | all_rows | 35,657 | 14.8% | 0.8203 | 0.8251 | +0.0048 | 0.2051 | 0.1993 | -0.0058 |
| de_train | mc_only | 5,290 | 100.0% | 0.8993 | 0.9019 | +0.0026 | 0.1507 | 0.1478 | -0.0029 |
| de_vertigo | all_rows | 33,363 | 15.0% | 0.7962 | 0.7969 | +0.0007 | 0.2128 | 0.2110 | -0.0018 |
| de_vertigo | mc_only | 4,995 | 100.0% | 0.8035 | 0.8132 | +0.0097 | 0.2017 | 0.1977 | -0.0039 |

**The headline.** The full ΔAUC range across all 18 (map × mode) cells is
**[-0.0056, +0.0114]**. The two largest positive deltas are `de_dust2`
mc_only (+0.0114) and `de_vertigo` mc_only (+0.0097). The two largest
negative deltas are `de_anubis` mc_only (-0.0056) and `de_overpass`
all_rows (-0.0054). Even the largest positive delta is roughly the
fold-to-fold standard deviation of the baseline model's AUC; we do not
treat any single ΔAUC as significant.

**Honest preliminary read.** In our first scalar-level check, BFS map
control is **largely redundant with `alive_diff`** once `alive_diff` is
already in the model. We do not interpret this as ruling out the
informativeness of map control under more specific scenarios — we read it
as a statement about the *form* of the feature we tested.

**What this experiment does NOT test.** Important limitations of this
ablation that future work needs to address:

- **Per-scenario slices.** We did not split the evaluation by post-plant
  vs. pre-plant, eco vs. full-buy, contested vs. uncontested, last-alive
  vs. multi-alive. The hypothesis that map control adds information in
  these slices specifically (and is washed out in the average) is the
  most natural alternative to the current scalar-redundancy story.
- **Specific maps.** Even within this table, `de_dust2` mc_only and
  `de_vertigo` mc_only show small positive ΔAUC; we have not attempted
  to characterize whether those are noise or whether dust2 / vertigo
  geometry produces structurally different MC signal.
- **Alternative MC formulations.** We tested only the polygon-area-weighted
  scalar in `[0, 1]`. Unscaled tile counts, multi-channel encodings (CT
  pure / contested / T pure / sightline-gated), per-region scalars
  rather than a single global scalar, and learned compositions of these
  primitives are all plausible alternatives. None of them are tested here.
- **Counterfactuals at the place level.** A candidate experiment is "does
  conditional MC at the bombsite, not global MC, predict post-plant CT
  retake outcome better than `alive_diff`-on-site?" The current ablation
  cannot distinguish that.

The bridge is to [`./FUTURE_WORK.md`](./FUTURE_WORK.md), which lists
these slices and feature redesigns in priority order.

## 8. Implementation-correctness validation

Distinct from the WP modelling, we wanted a separate empirical check that
the BFS implementation itself is computing what we believe it is — that
is, that the test suite passing is not concealing a structural bug.

**All 9 competitive maps smoke-tested.** The visualization driver at
[`scripts/generate_visualizations.py`](../../scripts/generate_visualizations.py)
runs the BFS pipeline on a synthetic frame for each of the 9 competitive
maps and writes both a heatmap and a "with players" overlay. The outputs
are in `docs/examples/map_control_validation/` (e.g.
`de_dust2_frame_heatmap.png`, `de_dust2_frame_with_players.png`). Visual
inspection confirms the BFS expansion respects the navmesh and player
locations are colored by the side that controls the tile they are
standing on.

**Real-demo runs.** The driver at
[`scripts/analyze_real_demo.py`](../../scripts/analyze_real_demo.py)
processes a parsed CS2 demo, filters out knife rounds and warmup ticks
(via length-of-round + round-state heuristics; see
`is_valid_competitive_round`), and runs the BFS frame metric across
every valid competitive round. We exercised this on Aurora vs Heroic
dust2 from the BlastTV Austin Major 2025 archive; the timeline of CT
control across the round (output at
`docs/examples/map_control_validation/de_dust2_real_timeline.png`) is
tactically consistent — CT control starts near 1.0 at freeze end, drops
sharply at the moment of T executes, and rebounds on retakes. The frame
overlay at `de_dust2_real_frame_players.png` and the round GIF at
`de_dust2_real_round.gif` show the BFS output on top of player positions
in the actual tactical context.

**Tile-to-place coverage on dust2.** A separate question we wanted to
nail down was whether every navmesh tile maps to a named place
(`Apartments`, `LongA`, etc.) so that downstream place-level analyses can
use BFS output without dropping tiles that fall in the gaps between named
regions. Aggregating over 10 dust2 demos gives 100% tile-to-place
coverage across 24 named regions, with BFS gap-fill closing the few
edge-case tiles that the static `nav_place_names` table did not cover.

**Bug fixes during validation.** Two issues surfaced during this work
that are worth recording:

- *`armor` column fallback.* The v2 schema requests `armor_value` from
  `demoparser2`, but the column actually returned is `armor`. Earlier
  real-demo analysis code was reaching for `armor_value` and failing
  silently; we patched the consumers to fall back to `armor` when
  `armor_value` is absent.
- *ZeroDivisionError in `_calculate_map_control_frame_metric`.* The
  metric divided total CT-controlled area by total controlled area
  without guarding the denominator. In the rare frame where every player
  is dead (e.g., the tick immediately after a 5v0 wipeout in a
  freeze-end edge case) the denominator is zero and the metric raised.
  Fixed to return `0.5` (neutral) in that degenerate case, matching the
  scalar's null-fill convention used elsewhere.

**Visual review interface.** The HTML at
`docs/examples/map_control_validation/index.html` collects the heatmaps,
overlays, and the dust2 timeline + GIF in one page for fast visual
review. This was the artifact we used to convince ourselves end-to-end
that the BFS is computing something sensible on real demo data, not just
on synthetic test frames.

The validation entry point for the numeric output side is
[`scripts/validate_results.py`](../../scripts/validate_results.py),
which checks that all materialized BFS metrics are in [0, 1], free of
NaN / Inf, and that the per-match means fall in a reasonable
[0.1, 0.9] range — flagging anything outside that as suspicious.

## 9. What is still open

This is a preliminary log; the negative result in §7 is best understood
as a starting point. The questions we believe are most important and
have not yet investigated:

- **Per-scenario ablation slices.** Repeat the §7 ablation conditioned on
  post-plant rounds, eco rounds, contested chokepoints, last-alive states,
  and per specific map. The current global-mean ΔAUC is the wrong unit of
  measurement if MC matters in slices and is uninformative on average.
- **Alternative spatial encodings.** Replace the single polygon-area
  scalar with: unscaled tile counts; a multi-channel
  CT-pure / contested / T-pure / sightline-gated representation; per-place
  scalars rather than one global scalar; structural / topological
  features (cuts, bottlenecks). Re-run the ablation with each.
- **Sightline-gated BFS validation.** The current BFS expansion is
  conical but does not use full sightline gating. Validate the
  `VisibilityChecker` integration on real demos and ablate.
- **Smoke / molotov / flash modifiers.** The BFS implementation has
  hooks for grenade-effect modifiers that we have not yet exercised on
  the full corpus; they may be the place where MC carries the
  most-orthogonal signal, since alive_diff cannot capture utility usage.
- **Longitudinal stability.** Does MC information content change as a
  match progresses (early rounds vs. late rounds), as economic state
  shifts (force-buys, eco rounds), or as the meta drifts across a
  tournament? Any of these would point to a more sophisticated
  per-state model rather than a flat global classifier.
- **Per-map sensitivity.** The §7 deltas vary across maps in a pattern
  worth interrogating (`de_dust2` and `de_vertigo` mc_only positive,
  `de_anubis` and `de_overpass` negative). Per-map BFS calibration is
  one of the least-investigated levers.

These directions are tracked in detail in
[`./FUTURE_WORK.md`](./FUTURE_WORK.md), with the conceptual framing
elaborated in [`./RESEARCH_VISION.md`](./RESEARCH_VISION.md).
