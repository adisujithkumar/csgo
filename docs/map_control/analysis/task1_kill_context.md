# Task 1 — Kill context & outcome

Generated from output/full_v2/cs2_analytics.duckdb (122,704 non-team kills covered).

## Distribution
| kill_context | kills | attacker_wr | avg_time_pct |
| --- | --- | --- | --- |
| advantage | 44707 | 0.958 | 0.777 |
| disadvantage | 36534 | 0.309 | 0.671 |
| parity | 39974 | 0.746 | 0.570 |

## Context x attacker_side
| attacker_side | kill_context | kills | attacker_wr |
| --- | --- | --- | --- |
| ct | advantage | 27005 | 0.949 |
| ct | disadvantage | 15836 | 0.258 |
| ct | parity | 21239 | 0.701 |
| t | advantage | 17702 | 0.973 |
| t | disadvantage | 20698 | 0.348 |
| t | parity | 18735 | 0.797 |

## CT swing analysis
| kill_context | ct_kills | ct_win_rate | avg_time_pct | avg_adv_before |
| --- | --- | --- | --- | --- |
| advantage | 27005 | 0.949 | 0.791 | 1.709 |
| disadvantage | 15836 | 0.258 | 0.669 | -1.502 |
| parity | 21239 | 0.701 | 0.575 | 0.000 |

## Top victim_place (CT kills, n >= 30)
| map_name | victim_place | n_kills | ct_win_rate | avg_time_pct |
| --- | --- | --- | --- | --- |
| de_overpass | Fountain | 33 | 0.909 | 0.551 |
| de_overpass | Construction | 54 | 0.889 | 0.834 |
| de_nuke | Hell | 63 | 0.841 | 0.757 |
| de_overpass | LowerPark | 200 | 0.835 | 0.742 |
| de_overpass | Canal | 98 | 0.816 | 0.676 |
| de_nuke | Vents | 105 | 0.810 | 0.758 |
| de_nuke | Control | 570 | 0.809 | 0.595 |
| de_mirage | Apartments | 970 | 0.801 | 0.717 |
| de_nuke | Hut | 627 | 0.801 | 0.690 |
| de_vertigo | Pit | 74 | 0.797 | 0.639 |