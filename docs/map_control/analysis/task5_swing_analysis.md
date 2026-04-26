# Task 5 — Map-control swing vs kill context

Sample size: 6,350 kills with before/after BFS pairs (from map_control_ticks_sample.parquet)

## Swing magnitude by context
| attacker_side | kill_context | n | mean_swing | median_swing | std_swing | attacker_wr |
| --- | --- | --- | --- | --- | --- | --- |
| ct | advantage | 1309 | 0.079 | 0.051 | 0.156 | 0.952 |
| ct | disadvantage | 878 | -0.018 | -0.010 | 0.144 | 0.233 |
| ct | parity | 1132 | 0.071 | 0.036 | 0.192 | 0.693 |
| t | advantage | 978 | 0.111 | 0.090 | 0.164 | 0.973 |
| t | disadvantage | 1062 | 0.016 | 0.014 | 0.132 | 0.363 |
| t | parity | 991 | 0.014 | 0.014 | 0.193 | 0.802 |

## Correlation: swing vs attacker_team_won
| kill_context | n | corr_delta_win | mean_delta | win_rate |
| --- | --- | --- | --- | --- |
| disadvantage | 1940 | 0.013 | 0.001 | 0.304 |
| parity | 2123 | 0.034 | 0.045 | 0.744 |
| advantage | 2287 | 0.054 | 0.093 | 0.961 |