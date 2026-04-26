# Research Vision — Map Control as a Spatial Object

> Companion documents: implementation [`./README.md`](./README.md), preliminary
> findings [`./ANALYSIS.md`](./ANALYSIS.md), tiered roadmap [`./FUTURE_WORK.md`](./FUTURE_WORK.md).
> This document is the long-form depth on the most ambitious research directions;
> `FUTURE_WORK.md` is the index of those directions plus the near-term work.

## 1. Premise

Counter-Strike 2 generates an unusually rich substrate for spatial-temporal
machine learning. Every match produces ten player trajectories sampled at
sixty-four ticks per second, yaws and pitches, hit-points, weapons, equipment
values, grenade detonations, kill events, bomb interactions, and round
outcomes. The map underneath those trajectories is a navigation mesh of two-
to-three thousand polygonal tiles with explicit adjacency, a visibility graph
materialized through ray-casting against `.tri` collision meshes, and a
human-readable place layer of roughly twenty-four named regions per
competitive map. Round outcomes provide a clean supervised label. Patches,
roster changes, and meta shifts impose distribution drift on a timescale we
can resolve. Pro and tier-two events provide hundreds of thousands of demos.
There are very few publicly available domains with this combination of
physical structure, dense temporal data, heterogeneous unit types, and ground
truth.

We are interested in *map control* as a research object inside this domain.
Map control is a tactical concept — at any moment of a round, which team
"controls" which parts of the map — that has a long history in the CS analytics
community and a concrete operational definition: from each alive player, run a
breadth-first search over the nav mesh; assign each tile a value that decays
with hop distance and (optionally) with angular deviation from the player's
yaw and weapon-class-conditioned cone; combine across the team; reduce to a
ratio CT / (CT + T) per tile. The current implementation lives in
[`awpy/stats/map_control.py`](../../awpy/stats/map_control.py), with a directional
weapon-aware extension already wired in via [`MapControlConfig`](../../awpy/stats/map_control.py)
and the conical BFS path. This is a credible spatial signal: when we average it
across the map area-weighted, we get a single number per frame on $[0, 1]$
representing "who controls more right now."

The companion analysis document records a preliminary scalar-level finding:
when we use that area-weighted scalar as a feature for win-probability
prediction, on top of a baseline that already includes alive-count differential,
the additional area-under-curve gain is approximately zero. This is not a
disproof of map control. It is a disproof of one specific use of it — collapsing
the entire spatial structure to one number per frame — and that disproof is
unsurprising in retrospect. The scalar reduction throws away exactly the part of
the data that makes CS interesting: a team with twenty-five percent of the map
in the right twenty-five percent (mid plus B apartments before a hit) is in a
very different position from a team with fifty percent of the map in the wrong
fifty percent (long plus pit during a default). The scalar treats those as if
the second is twice as good.

The research bet of this document is that the spatial structure becomes
predictive in ways scalar counting cannot match, *provided* we encode it well
and pair it with the right downstream tasks. We organize the bet along two axes.
The first axis is representation: what is the right encoding of "control" — raw
nav-mesh tile graph, rasterized image, place-level pooling, hierarchical
mixture? The second axis is usage: what tasks should that encoding feed —
win-probability prediction, kill-outcome prediction, behavioral cloning of pro
movement, strategy clustering, meta-shift detection, cross-map generalization?
The five directions below sweep the cells of that two-by-five grid we find
most interesting. Each direction has a stated motivation, a sketch of the
approach, the open questions we cannot currently answer from a chair, and a
*concrete first experiment* that would either generate the next interesting
problem or rule the direction out cheaply.

We try to be honest about cost: many of these directions require GPU
training and well-conditioned data at a scale we have only partially built. The
ETL pipeline at [`scripts/etl_pipeline.py`](../../scripts/etl_pipeline.py) is the
foundation; we will reference it where the research direction depends on
specific tables. Section 8 collects the structural limitations that constrain
what is plausible regardless of how clever the model is.

## 2. Direction 1 — Multi-channel spatial encoding

### Motivation

The single-scalar map-control metric was derived to support broadcasts and
post-game review. It is summary statistics, and it discards by design the
structure that supervised learners would want to consume. The opposite
extreme — handing the model the raw nav-mesh graph with one scalar per tile and
letting message-passing figure it out — is also bad: with two thousand tiles, a
three-hop graph convolution sees one to two percent of the graph, and CS
relevant scales (mid-to-A, T-spawn-to-mid) span ten or more hops. The middle
ground we want to occupy is a *thick per-tile feature vector*, where each tile
carries enough channels that a moderately-deep network — convolutional,
graph-attentive, or transformer — can learn the interactions between channels
without first having to invent the channels.

### Encoding sketch

For each frame, every nav tile gets a feature vector of roughly twenty-one
channels. The channels we believe are necessary, with their motivation:

- **Per-team active control depth (2 channels)**: the BFS-derived control
  contribution from CT and T, separately. Keeping them separate rather than as
  a ratio preserves the difference between "neutral because nobody is here"
  and "neutral because both teams pressure equally."
- **Per-team alive presence (2 channels)**: indicator-like channels for
  "an alive player from this team is *on* this tile" plus a smoothed kernel
  for "is near." Distinct from BFS contribution because BFS values include
  hop-decayed projections; presence is the source.
- **Per-team mean active equipment value (2 channels)**: economy projected onto
  the tile via the same BFS; tells the model whether the control is being
  exerted by an AWPer or by a pistol round eco.
- **Per-team weapon-class composition (6 channels)**: one channel per class in
  `WEAPON_CLASSES` (awp, rifle, smg, pistol, shotgun, knife) carrying the
  fraction of the team's projected control onto this tile that comes from each
  class. Lets the model represent "this tile is controlled by an AWP cone" vs
  "this tile is controlled by SMGs in close quarters" without the model having
  to infer it.
- **Per-team cone reach (2 channels)**: forward-cone-only contribution from
  the conical BFS, separately by team. Distinguishes facing from presence.
- **Per-tile flash duration (1 channel)**: the maximum residual flash time
  among players attempting to read the tile, decayed appropriately.
- **Per-tile smoke / molotov denial (2 channels)**: indicator that a smoke
  occludes line-of-sight through this tile, plus a continuous channel for
  molotov / inferno denial pressure.
- **Bomb-zone proximity (1 channel)**: hops to nearest bombsite tile, clipped.
- **Time-to-end (1 channel)**: seconds remaining in the round, broadcast onto
  the tile (scalar broadcast, but kept as a channel so the conv stack can
  modulate spatial features by phase).
- **Place identity (1 channel)**: a learned embedding of the tile's place
  (Long, Catwalk, A site, ...). For an initial experiment, a one-hot expansion
  into roughly twenty-four extra channels also works; we prefer learned
  embeddings as soon as the place hierarchy is plumbed end-to-end.

The choice we want the model to *learn* and not us to *engineer* is the
interaction between these channels: e.g. a high CT depth × high CT AWP-class
fraction × high pitch-into-mid is a different tactical situation from a high
CT depth × high CT pistol fraction × moderate mid pressure. A linear scalar
collapse cannot represent that interaction. A multi-channel feature vector
plus a learned interaction layer can.

### Architecture options

We see three reasonable architecture families, each with a trade-off:

1. **Rasterize-and-convolve.** Voxelize the playable map into a grid (say
   $128 \times 128$ on the radar scale, with a binary mask for off-map cells)
   and project per-tile features onto the grid using either Voronoi
   assignment from tile centroids or rasterization of tile polygons. Then run
   a 2D convolutional stack — ResNet-12 or Swin-tiny scale — and read out via
   a global-pool head for win-probability or a per-pixel head for tile-level
   tasks. *Strength*: rotation-equivariant pooling layers come for free; tons
   of pre-trained image work to copy. *Weakness*: convolutions assume planar
   topology, but the nav mesh is not planar — bridges, ramps, vents, and
   stacked floors (the entire upper-vs-lower nuke problem) violate the
   assumption. We already had to introduce a Z-floor gate in the conical BFS
   to correct for the equivalent bug. Rasterization will silently re-introduce
   that bug unless we use a multi-channel Z-aware encoding (e.g. one channel
   per Z bin) or accept the corruption.

2. **Treat tiles as graph nodes.** Build a graph where each node is a tile,
   edges are the nav adjacency `connections`, and node features are the
   twenty-one channels. Apply a graph attention network with several layers,
   pool to a frame-level vector via attention, classify. *Strength*: respects
   the actual topology of the map, including vertical structure. *Weakness*:
   shallow GNNs (the only practical depth before over-smoothing) see only
   their few-hop neighborhood; useful CS interactions span much farther. Also,
   GNN pooling over two thousand nodes with twenty-one channels is reasonable
   but not free at training time.

3. **Treat hierarchical regions as tokens.** Pool tile features up to per-place
   features (twenty-four tokens × twenty-one channels per map), pass through
   a small transformer with attention over places, then pool with a learned
   `[CLS]` token. *Strength*: the token count is small, attention is
   $O(N^2)$ but $N \approx 24$, and *places* are the granularity humans
   reason at. *Weakness*: per-place pooling discards within-place spatial
   structure (an attacker pushing pit is on A site, but so is a defender
   holding default). We address this with Direction 2.

We expect the best practical model to be a hybrid: rasterize for the local
context the convolutional inductive bias buys us, but also pool to place tokens
for the long-range attention. The first experiment below picks the simplest
member of the family to establish a floor.

### Open questions

- **Supervisory signal.** Round outcome is the cleanest label but is sparse: a
  best-of-three averages forty to sixty rounds, and label noise from "won despite
  bad position" is high. Kill outcome is denser (≈eight per round) but is
  about kills, not control. A value head trained against round outcome with
  per-tick TD updates is the intermediate choice and is what we currently lean
  toward; we want to compare to a model trained on per-kill outcomes (which
  team gets the next kill) as an auxiliary head.
- **How much does multi-channel actually beat scalar?** The headline question.
  Once `alive_diff` is in the baseline, the scalar map-control metric adds
  approximately nothing. We need to measure how much the twenty-one-channel
  representation adds, ideally with channel-ablation studies (drop the
  weapon-class channels and see what happens).
- **Robustness to map changes.** Valve patches occasionally change geometry
  (the Mirage A-site rework, the Inferno banana redesign). A model that has
  baked in the place graph as fixed will need re-training per patch. A more
  robust formulation conditions on the nav mesh structure itself, which moves
  us toward Direction 5.

### Concrete first experiment

Implement the channel pipeline as a nightly Polars-to-Parquet job: for each
tick we emit a `(n_tiles, 21)` array plus the round-level label and the
alive-count differential. Train a small 2D conv net (rasterize tile centroids
via Voronoi, eight channels reduced via $1 \times 1$ conv, ResNet-12 backbone,
sigmoid head) to predict per-frame round-win probability. Compare to the
existing `train_wp_model_v2.py` baseline that uses scalar
`alive_diff + map_control_scalar`. Report ΔAUC on a held-out set of demos
*and* a SHAP-style per-channel importance via integrated gradients, so we can
see which channels are doing the work. If ΔAUC over scalar is below a defensible
threshold (say one absolute AUC point), we have learned that the rasterization
choice itself is wrong and Direction 2 should be tried first; if ΔAUC is above,
we have established a thicker baseline to build on.

## 3. Direction 2 — Hierarchical attention over places

### Motivation

Humans do not reason about CS at tile granularity. A coach does not say
"player six should rotate from tile 1843 to tile 1247." They say "rotate from
banana to mid." Places are the natural unit of abstraction: there are roughly
twenty-four per competitive map (we have validated 100 percent tile-to-place
coverage on dust2 across ten demos), they are stable across patches except
when sites are reworked, they are how broadcasters call rounds, and they
correspond to the strategic vocabulary teams already use. The *hypothesis* is
that learning at place granularity will generalize and interpret better than
either tile-level or scalar models.

### Approach sketch

The pipeline has three stages.

1. **Tile-to-place pooling.** Given the per-tile twenty-one-channel feature
   vector from Direction 1 and the place mapping from
   [`docs/examples/map_control_validation/tile_to_place_dust2_full.json`](../../docs/examples/map_control_validation/tile_to_place_dust2_full.json),
   pool tiles to places. The naive choice is mean pooling. A better choice is
   attention pooling, where each tile produces an attention weight (a function
   of its features), and the place vector is the weighted sum. Attention
   pooling lets the model decide that, in a place like A site, only the pit
   tiles matter for a default-vs-execute distinction.
2. **Place-token transformer.** Treat the resulting per-place vectors as
   tokens. Build a transformer with four to six layers, eight attention heads,
   and a learned positional encoding derived from place adjacency in the nav
   mesh (so attention has a structural prior to start from). Pool with a
   learned `[CLS]` token.
3. **Round-level temporal stack.** Wrap the per-frame place transformer in
   either an LSTM over the timeline or a temporal transformer. The natural
   sub-frame breakpoints are `freeze_end`, every kill, every plant, and every
   utility detonation; sampling at those breakpoints rather than every tick is
   a big computational win and probably a small accuracy loss. The temporal
   stack reads from the per-frame `[CLS]` embedding.

### The hierarchy advantage

At early-round, places are coarse: a model probably wants to know
"are CTs leaning B?" and place-level attention is sufficient. At mid-round,
places are right: contests happen at named regions. At late-round, especially
in retake situations, the right granularity is *tile*: who exactly is holding
which corner of the bombsite. We propose an *adaptive* attention pattern via
a learnable gate that, conditioned on time-remaining and alive-count, chooses
how much to weight place-level vs. tile-level features. This is similar to
multi-scale pyramid pooling in dense-prediction vision: different scales for
different round phases.

### The "why ~24 tokens" argument

Twenty-four tokens is small enough that attention can be deep without blowing
up compute, learning patterns of *place co-activation* that would be
unrecoverable with full tile attention (which would have to discover the
place structure from data). Twenty-four is also large enough to disambiguate:
a "mid + B apt + B site" co-activation is recognizably different from a
"long + A pit" co-activation. By comparison, a four-token quadrant model could
not separate "double mid" from "split mid + A," which are tactically distinct.

### Open questions

- **Generalization across maps.** The place names and adjacencies differ per
  map. A naive transformer with place-id-embedding-as-token cannot transfer.
  Two options: (a) per-map model with shared backbone; (b) inject place
  *features* (size, distance to bombsites, connectivity) instead of place
  *id*, and let the attention learn over a structural representation. The
  second option is harder and is what Direction 5 explores.
- **Patch resilience.** A site rework changes both place geometry and place
  adjacency. The shared-backbone option requires re-fitting the place
  embeddings; a structural-feature option might survive without re-fitting.
- **Attention interpretability.** A central reason to want this architecture
  is that the attention maps are *legible*: you can plot which place tokens
  attend to which other place tokens at each round phase, and a coach can read
  the answer. Whether the attention recovers human-recognizable strategies
  (default A vs. fast B execute vs. mid take) is testable.

### Concrete first experiment

Aggregate per-tile control to per-place vectors using mean pooling (skip the
attention pool for the first pass — we want to know if the structure helps
before we ask how it helps). Train a four-layer, eight-head transformer with
hidden size 128, conditioned on the per-frame twenty-four-by-twenty-one input
plus a global time-remaining and alive-diff scalar. Predict round outcome.
Compare AUC against the Direction-1 raster CNN. Then, on a held-out set of
fifty rounds where we have manually labeled the macro tactic (default A,
default B, fast A, fast B, mid take, eco rush, retake), pull the
attention maps at the kill events and compare to the labels. We are looking
for two things: a measurable AUC delta, and a recognizable mapping from
attention patterns to strategies. If the AUC delta is positive but small and
the attention maps are illegible, the answer is "the model is finding
something but not the human concept"; if the maps look legible, we have an
interpretability handle for free.

## 4. Direction 3 — IGL navigation model

### Motivation

The most ambitious idea in this document. CS at the highest level is decided
by the In-Game Leader's call set: where to position five players, when to
trade information for space, when to fake, when to commit. Modeling that role
machine-learning-style is a very large bet, but a tractable one: we have
hundreds of thousands of pro demos as imitation targets, the round-outcome
label gives us a reward signal, and the problem decomposes into a sequence of
discrete-ish decisions (where should each player be at each freeze-end and at
each post-trade re-position?) rather than a continuous control problem. We
think it is worth attempting because the data is unusually plentiful for what
is, structurally, an offline reinforcement learning problem on a domain with a
clean reward signal.

### The bet, in concrete terms

We want to train a policy
$\pi(a_t \mid s_t)$ where $s_t$ is the per-frame state encoding from
Directions 1 and 2, and $a_t$ is, for each CT player, a target *place*
distribution they should be heading toward over the next $N$ seconds. We
choose place rather than tile because (a) tile-level supervision from pro
demos is sparser per state than place-level, and (b) place is the natural
unit of human play. Conditioning is on economy, score line, prior round
outcomes, recent kills (positions and weapons), and the current frame state.

### Two-stage training

1. **Behavioral cloning.** Take pro demos. For each `freeze_end` and each
   subsequent positional waypoint (defined as a stable place residency lasting
   at least, say, two seconds), record the (state, action) pair where the
   action is the place the player ended up in. Train the policy with negative
   log-likelihood. This captures pro positional tendencies and gives us a
   strong prior — pros are good — but it is bounded above by pro performance
   and has no notion of *why* the position was chosen.
2. **Win-probability fine-tuning.** Use the value head from Directions 1
   and 2 (trained against true round outcome) as a critic. Run a policy
   gradient — Advantage Weighted Regression is a reasonable simple choice
   here, since it avoids on-policy rollouts and is well behaved for offline
   data — to push the policy toward actions whose resulting state has higher
   estimated WP. This gives us a way to *exceed* the BC prior in cases where
   the prior is wrong (failed defaults that nonetheless got copied because
   the team won via individual skill, etc.).

### Architecture

The state encoder is the multi-channel + hierarchical-attention model from
Directions 1 and 2 (this is the integration point that justifies the earlier
work). Wrap it in an LSTM that runs along the round timeline. The action head
is a categorical distribution over the place list per CT player, with
state-conditioned masks (no point predicting "rotate to A site" if the player
has been dead for forty seconds — the mask is identity, but we keep the
formalism for clarity). Five player heads share the encoder backbone.

### Open questions

- **Does BC saturate before fine-tuning helps?** Empirically, BC on pro data
  often hits a performance ceiling at the level of the median pro player.
  Whether WP fine-tuning can clear that ceiling on held-out matches is the
  central empirical question.
- **Does the policy converge to recognizable strategies?** If we cluster the
  policy's outputs over many rounds (Direction 4 below), do the clusters look
  like human-named strategies — default, fast B, slow B, mid take, retake from
  CT spawn — or do they look like idiosyncratic learned behaviors that happen
  to win? Either answer is informative. The first is a sanity check; the
  second might be a *new* strategy.
- **Generalization across maps.** A policy trained only on dust2 will not
  transfer to mirage. A multi-map joint training with a map-conditioned
  encoder is the natural extension and is also Direction 5.
- **Sport-integrity considerations.** A working IGL policy is a coaching
  artifact. Making it available raises questions about competitive equity
  (does a team with the policy have an unfair advantage at lower tiers?), and
  about what to do with the policy's failure modes (a policy that predicts
  "always rotate to B at tick 7000" because that one team did it once will
  give very confident wrong answers). We commit to publishing only aggregate
  policy *clusters* and aggregate evaluation results for now, not the policy
  itself, until those questions have more thought.

### Concrete first experiment

Restrict to dust2 only, CT side only, BC only — no fine-tuning. Action space
is the twenty-four-place vocabulary. State encoder is the hierarchical-attention
model from Direction 2 (use the $[\text{CLS}]$ embedding plus per-player
features). Train on 80% of the available dust2 pro demos (roughly four hundred
matches × twelve to fifteen rounds per match), eval on the remaining 20%.
Measure top-1 and top-3 place-prediction accuracy at `freeze_end` and at
post-kill re-position events. Cluster the policy outputs over the eval set
into roughly five tactic groups using a Gaussian mixture on the LSTM final
state. Hand-label each cluster (we already have the manual-tactic labels from
Direction 2's experiment, repurpose them). If the cluster frequencies match
HLTV-noted tournament metas in the demo time range — for example, an uptick
in mid-take frequency in the wake of a notable team's mid-take meta in
mid-2024 — we have evidence the model has learned the macro layer. If the
clusters do not correspond to anything we recognize, we have learned that BC
alone isn't sufficient and the value-head fine-tune is required.

## 5. Direction 4 — Strategy clustering and meta-shift detection

### Motivation

Pro CS strategy is not stationary. Patches change weapon balance and map
geometry, anti-strats propagate via demo review, IGLs move teams, regional
metas diverge. A model that ignores time will be mis-specified. A model that
*tracks* time can do interesting things: feature-engineer "this round looks
like a fast B" as a side channel, monitor for a meta shift that should
trigger re-training, fingerprint a team by its strategy mix.

### Approach

We treat each round as a time series of place-occupancy vectors — at each
sample, the fraction of (alive) team that is in each place. With twenty-four
places and one round being roughly forty samples (every two seconds for ~80
seconds of action), each round becomes a $40 \times 24$ matrix.

Cluster these matrices with dynamic time warping as the distance metric and
HDBSCAN as the clusterer. DTW is the right metric because rounds vary in
length (a fast hit takes thirty seconds, a slow default takes ninety) but
have the same place trajectories. HDBSCAN tolerates noise and produces a
density-based clustering that does not require us to pre-specify the number
of clusters — a useful property given that tactic counts are not constants.

For meta-shift detection, run change-point detection — Bayesian Online
Changepoint Detection or a simpler PELT formulation — on the per-cluster
frequency time series at the patch and event level. Patches and events are
known timestamps; the question is whether the frequency change is large
relative to the patch-to-patch noise.

### Use cases

- **Feature engineering.** Cluster id (or soft cluster posterior) becomes
  another input channel for the WP and IGL models. A WP model conditioned on
  "this is a fast B" predicts late-round outcome better than a model that has
  to infer the same from raw position.
- **Meta-monitoring.** When the per-cluster frequency time series
  changes, that is a signal to retrain downstream models. Most ML systems do
  this via blanket calendar retraining; cluster-frequency change-point gives
  a principled trigger.
- **Team-style fingerprinting.** Per team, compute the empirical
  distribution over clusters across their match history. Two teams that play
  the same map differently (e.g. one heavy on mid-take, one heavy on B site
  default) will have distinct cluster vectors, and the divergence between
  them is a clean style-similarity metric.

### Concrete first experiment

Restrict to dust2 CT side. Define round trajectory as a $T \times 24$ matrix
of place occupancies. Compute pairwise DTW distances on a five-thousand-round
sample (this is plausible on a single workstation in a few hours), feed to
HDBSCAN with `min_cluster_size = 50`. Inspect the resulting clusters by hand:
do the centroids look like default A, default B, fast B, mid take, retake?
Then, on a held-out two-thousand-round set with timestamps, plot the
per-cluster frequency time series across major patches in the data range,
and run a change-point detector. We are looking for change points that
co-occur with patch dates rather than uniformly. This experiment is cheap
relative to the others, and even a negative result is informative — it tells
us whether DTW + HDBSCAN is the right substrate or whether we need a learned
distance.

## 6. Direction 5 — Map-conditioned generalization

### Motivation

Everything above is per-map. The deeper question — whether a single model can
encode CS knowledge that generalizes across maps — is the most ambitious
research bet of the document. If the answer is yes, training on the seven
to nine current competitive maps should benefit a hold-out map, including
maps Valve has not yet released. The standard approach to multi-task learning
suggests this should help even modestly: representational sharing usually
helps when tasks share structure, and CS maps share a lot — bombsites,
default positions, choke points, AWP angles. The standard counter-argument is
that map-specific knowledge dominates: knowing that the dust2 long doors clip
matters less for inferno banana than the *general* idea of "narrow choke."

### Sketch

Two architectural moves.

First, *condition on a map embedding*. Each map gets a learned vector
embedding, fed to the state encoder as a side channel. This is the multi-task
learning baseline: parameters are shared, but the embedding lets the model
specialize. It works as long as the map embedding can carry enough
information to differentiate maps. We expect this to outperform single-map
models on hold-out match prediction within the trained-on map set, but to
fail to transfer to a new map without re-fitting the embedding.

Second, *meta-learn the map embedding*. Train the embedding parameters via a
second-order objective (MAML-style or first-order MAML / Reptile, given
compute constraints): each task is "predict round outcome on map M with
embedding E_M," and the meta-objective is to make the joint $(E_M,
\text{shared params})$ such that fine-tuning $E_M$ on a few hundred rounds
of a *new* map suffices for good performance. This is the bet on transfer.

Third, the more radical option: drop map identity entirely and condition only
on a *graph-isomorphism-friendly* representation of the nav mesh. The
encoder reads (a) the nav mesh structure as a graph, (b) the place hierarchy,
(c) per-tile features. If the model is well-formed under graph isomorphism,
two structurally-equivalent map regions on different maps should encode
identically. This is hard but is the right limit of "map-agnostic CS
knowledge."

### Open questions

- **How much CS knowledge is universal?** We do not know. Default positioning
  patterns probably are; specific trade-frag setups probably are not.
- **Does the meta-learned embedding help on the next Valve map?** This is
  the test we cannot run today because Valve has not released the next map.
  The proxy test is leave-one-map-out: hold out mirage, train on the rest,
  fine-tune the mirage embedding on a small held-out training portion of
  mirage data, and measure performance on the rest of mirage. Compare to a
  from-scratch mirage-only baseline.
- **Player-pose invariances.** Yaw and pitch are continuous and rotational.
  A truly map-agnostic model should be invariant under arbitrary rotation of
  the map (if we rotated dust2 ninety degrees, the same calls should still
  win). Architectures that build in $\mathrm{SE}(2)$ equivariance (group
  convolutions, equivariant transformers) might generalize better, at the
  cost of complexity. We list this as an open question but not a first
  experiment.

### Concrete first experiment

Train the Direction-1 raster-CNN model jointly on five maps with a learned
$d=16$ map embedding fed as a side channel. Hold out one map (mirage).
Compare hold-out AUC on mirage to (a) the from-scratch single-map mirage
baseline, (b) the same model with the embedding initialized to the mean of
the trained embeddings rather than learned. The test we want to pass: the
joint model with mean-embedding initialization is within one AUC point of
the trained baseline. If yes, we have evidence of representational sharing.
If no, we need to re-think the architecture before investing in meta-learning.

## 7. Cross-cutting research questions

The five directions are not orthogonal; they probe four shared questions that
we will keep returning to:

- **Where is the right level of abstraction for control?** Tile, place,
  quadrant, scalar. The scalar fails (this is Section 1). The tile-level
  is too sparse for shallow models and too dense for deep ones unless we
  pre-aggregate. The place-level seems most promising. The quadrant-level
  is a useful sanity check (a four-token model is the trivial baseline above
  scalar).
- **Does the abstraction depend on the task?** Round-outcome prediction
  rewards aggregating information; policy learning rewards keeping it.
  Strategy clustering benefits from temporal trajectories; meta-shift
  detection benefits from cluster posteriors. The right answer might be
  *different for each direction*, in which case a universal encoder is the
  wrong architecture and we should accept per-task encoders that share a
  pretrained backbone.
- **How much do data scale and quality matter?** Pro demos overrepresent
  HLTV-stars-filtered tournament games. Tier-two events have different
  meta and different player skill. A model trained only on the top
  fifteen teams might not transfer to FPL-tier matches, and *vice versa*.
  Whether any of these directions changes its answer at $10\times$ data is
  a critical empirical question we cannot answer until the ETL pipeline
  fully ingests the 906-RAR dataset.
- **What is the simplest model that beats the alive-diff-plus-scalar
  baseline by a defensible margin?** The most rigorous version of all of
  the above. Until we have an experimentally-validated answer, the rest of
  the program is partly speculation. Direction 1's first experiment is
  designed to be that test.

## 8. Constraints and pitfalls

Honesty about what would make this hard.

The demoparser2 schema has occasional gaps. Some fields (notably `armor` in
some demo paths) are inconsistently present, which is why the alive filter
in [`awpy/stats/map_control.py`](../../awpy/stats/map_control.py) falls back to
`health > 0` when armor is missing. Multi-channel encoding inherits these
gaps: any channel that depends on `current_equip_value`, `flash_duration`,
or `weapon` is at the mercy of demoparser2 emitting it correctly across
versions. We expect to spend non-trivial engineering effort on imputation
and validation as the data scales.

Sixty-four-tick sampling is coarser than the action it samples. A flick peek
at twenty AWP-rounds-per-second can complete inside a single tick, which means
some tactically-significant events are not even visible to the model. A
research direction that needs sub-tick granularity (a "flash duel" model, for
example) is not supported by the data substrate. We accept this and orient
research around tactic-level structure rather than sub-tick mechanics.

The ETL pipeline at [`scripts/etl_pipeline.py`](../../scripts/etl_pipeline.py)
correctly filters knife rounds in some paths and incorrectly in others. We
are aware and have it on the near-term roadmap; a model trained on raw
Parquet output might pick up knife rounds and learn a bias toward "everyone
is in mid → CT wins," which is a knife-round artifact, not a real signal.
We pre-filter at experiment-script level to mitigate.

Pro demos overrepresent stars-filtered HLTV matches. The strongest teams
play a non-uniform map distribution (mirage and inferno are overrepresented;
ancient is under-represented relative to the current pool), and a stars-filter
biases toward higher-skill demos within the top tier. Models trained on this
distribution will be calibrated for high-tier play and may degrade gracefully
or sharply on tier-two events. We have not measured the degradation.

Map balance varies. Several competitive maps have a CT or T sided win-rate
asymmetry of three to seven percentage points. WP models trained without
side conditioning will pick this up as a signal, which is correct on the
training distribution but reduces interpretability — a "this side is winning"
prediction at $t = 0$ is partly the model reading off the side asymmetry,
not the position. We will condition on side-and-map at inference to disentangle.

Patches reset the meta. Every time a major patch lands, some fraction of
the map control we are reasoning about is invalid: a removed angle, a
re-routed boost, a different default position. Direction 4 (cluster
frequencies) will help us *detect* these shifts; it will not *predict* them.
A robust system needs continuous re-training infrastructure that we have
not built.

Computational budget for spatial-temporal models is real. A four-layer
place transformer with twenty-four tokens is cheap. The same model wrapped
in an LSTM over forty per-round samples × thousands of rounds × eight or
nine maps is non-trivial. The IGL policy fine-tune is non-trivial. We do not
yet have a costed-out compute estimate for the full direction set — that is
near-term planning work.

The ground-truth round outcome label is sparse. A best-of-three averages
roughly fifty rounds, and a tournament has roughly thirty matches. To get
to a hundred thousand labeled rounds we need most of the 906-RAR dataset
ingested. Direction 1's first experiment is sized to be runnable with about
ten thousand labeled rounds; later experiments scale up.

Sightline-gating depends on accurate `.tri` files. The current
`VphysParser` in [`awpy/visibility.py`](../../awpy/visibility.py) reads only
`m_parts[0].m_rnShape.m_hulls[...]` and the corresponding meshes; maps with
multiple physics parts have geometry silently skipped. The visibility check
in such a map will return "visible" for occluded segments, which is a false
negative on occlusion. Direction 1's smoke / molotov channel inherits this
limitation: smokes near multi-part geometry will under-occlude. We have it
on the roadmap; in the interim we use sightline-gating only on the maps
where `m_parts[0]` covers the static geometry of interest.

Finally — and this is the most important constraint — the scalar finding is
not an invitation to prove a different scalar is better. The research bet
is that the *spatial structure* matters; the way we cash in on that bet is by
preserving structure all the way through to the model. A direction that ends
in "we found a better single number" would be a partial failure even if the
number is better, because it would mean we failed to use the data the
collected demos contain.
