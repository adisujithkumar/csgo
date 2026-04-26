# Task 2 — round-time importance + place-value proxy

Sampled tick rows used: 181,927

## Decile-wise correlation with CT round win
| time_decile | n_ticks | corr_alive_diff | corr_hp_diff | corr_equip_diff | corr_is_bomb_planted |
| --- | --- | --- | --- | --- | --- |
| 0 | 20857 | 0.029 | 0.031 | 0.394 | nan |
| 1 | 18031 | 0.142 | 0.137 | 0.399 | nan |
| 2 | 17957 | 0.245 | 0.250 | 0.433 | nan |
| 3 | 18028 | 0.335 | 0.342 | 0.467 | -0.030 |
| 4 | 17892 | 0.405 | 0.413 | 0.506 | -0.095 |
| 5 | 17610 | 0.498 | 0.494 | 0.565 | -0.176 |
| 6 | 17916 | 0.589 | 0.579 | 0.619 | -0.318 |
| 7 | 17949 | 0.661 | 0.651 | 0.674 | -0.482 |
| 8 | 17807 | 0.735 | 0.724 | 0.731 | -0.500 |
| 9 | 17880 | 0.813 | 0.793 | 0.791 | -0.550 |

## Logistic regression of ct_won ~ alive_diff per decile
| time_decile | n_ticks | coef_alive_diff | intercept | brier_rmse |
| --- | --- | --- | --- | --- |
| 0 | 20857 | 0.627 | -0.132 | 0.499 |
| 1 | 18031 | 0.739 | -0.151 | 0.494 |
| 2 | 17957 | 0.829 | -0.169 | 0.484 |
| 3 | 18028 | 0.947 | -0.172 | 0.470 |
| 4 | 17892 | 1.012 | -0.158 | 0.456 |
| 5 | 17610 | 1.187 | -0.160 | 0.430 |
| 6 | 17916 | 1.360 | -0.074 | 0.396 |
| 7 | 17949 | 1.556 | -0.049 | 0.362 |
| 8 | 17807 | 1.844 | -0.205 | 0.315 |
| 9 | 17880 | 2.403 | -0.792 | 0.241 |

## Top victim_place — CT kills
| map_name | victim_place | n_kills | ct_win_rate_given_kill | avg_time_pct | pct_disadv |
| --- | --- | --- | --- | --- | --- |
| de_overpass | Construction | 54 | 0.889 | 0.834 | 0.241 |
| de_nuke | Hell | 63 | 0.841 | 0.757 | 0.190 |
| de_overpass | LowerPark | 200 | 0.835 | 0.742 | 0.235 |
| de_overpass | Canal | 98 | 0.816 | 0.676 | 0.173 |
| de_nuke | Vents | 105 | 0.810 | 0.758 | 0.143 |
| de_nuke | Control | 570 | 0.809 | 0.595 | 0.084 |
| de_mirage | Apartments | 970 | 0.801 | 0.717 | 0.143 |
| de_nuke | Hut | 627 | 0.801 | 0.690 | 0.134 |
| de_vertigo | Pit | 74 | 0.797 | 0.639 | 0.216 |
| de_train | BackofB | 557 | 0.797 | 0.656 | 0.169 |
| de_overpass | UpperPark | 81 | 0.790 | 0.676 | 0.198 |
| de_overpass | BombsiteB | 156 | 0.782 | 0.809 | 0.224 |
| de_nuke | Trophy | 182 | 0.775 | 0.550 | 0.181 |
| de_inferno | TRamp | 155 | 0.774 | 0.490 | 0.135 |
| de_train | BombsiteB | 943 | 0.767 | 0.782 | 0.223 |
| de_overpass | BombsiteA | 131 | 0.763 | 0.824 | 0.336 |
| de_train | TMain | 565 | 0.763 | 0.621 | 0.204 |
| de_inferno | TopofMid | 1073 | 0.761 | 0.681 | 0.194 |
| de_mirage | SideAlley | 88 | 0.761 | 0.665 | 0.193 |
| de_inferno | Underpass | 54 | 0.759 | 0.601 | 0.130 |

## Top victim_place — T kills
| map_name | victim_place | n_kills | t_win_rate_given_kill | avg_time_pct | pct_disadv |
| --- | --- | --- | --- | --- | --- |
| de_anubis | CTSpawn | 43 | 0.907 | 0.830 | 0.186 |
| de_anubis | CTSideUpper | 44 | 0.886 | 0.759 | 0.182 |
| de_anubis | Street | 60 | 0.850 | 0.801 | 0.200 |
| de_nuke | Vending | 104 | 0.827 | 0.623 | 0.212 |
| de_mirage | Shop | 357 | 0.826 | 0.789 | 0.283 |
| de_anubis | Alley | 246 | 0.821 | 0.802 | 0.285 |
| de_train | Entrance | 49 | 0.816 | 0.795 | 0.143 |
| de_inferno | Ruins | 429 | 0.809 | 0.801 | 0.326 |
| de_anubis | Ruins | 94 | 0.809 | 0.746 | 0.245 |
| de_dust2 | Middle | 240 | 0.808 | 0.650 | 0.242 |
| de_anubis | LowerTunnel | 41 | 0.805 | 0.781 | 0.220 |
| de_ancient | Outside | 187 | 0.802 | 0.666 | 0.225 |
| de_dust2 | OutsideLong | 77 | 0.792 | 0.660 | 0.221 |
| de_train | Ivy | 96 | 0.792 | 0.528 | 0.240 |
| de_anubis | Middle | 573 | 0.791 | 0.608 | 0.290 |
| de_dust2 | TunnelStairs | 76 | 0.789 | 0.524 | 0.276 |
| de_dust2 | UpperTunnel | 291 | 0.787 | 0.620 | 0.275 |
| de_anubis | PalaceInterior | 200 | 0.785 | 0.734 | 0.290 |
| de_nuke | Heaven | 280 | 0.782 | 0.761 | 0.250 |
| de_mirage | BackAlley | 45 | 0.778 | 0.720 | 0.333 |

## CT disadvantage-kill swing
| map_name | victim_place | n_advantage | n_disadvantage | n_parity | wr_advantage | wr_disadvantage | wr_parity | disadv_minus_parity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de_anubis | CTSpawn | 4 | 24 | 8 | 0.750 | 0.042 | 0.000 | 0.041666666666666664 |
| de_nuke | CTSpawn | 0 | 25 | 1 | 0.000 | 0.000 | 0.000 | 0.0 |
| de_mirage | Truck | 54 | 27 | 37 | 0.963 | 0.407 | 0.595 | -0.18718718718718724 |
| de_vertigo | ARamp | 32 | 26 | 26 | 0.969 | 0.500 | 0.731 | -0.23076923076923073 |
| de_nuke | Trophy | 75 | 33 | 74 | 0.947 | 0.485 | 0.730 | -0.24488124488124485 |
| de_overpass | LowerPark | 87 | 47 | 66 | 0.966 | 0.574 | 0.848 | -0.2740167633784655 |
| de_anubis | Street | 16 | 32 | 17 | 0.938 | 0.250 | 0.529 | -0.27941176470588236 |
| de_inferno | Library | 11 | 22 | 7 | 1.000 | 0.136 | 0.429 | -0.2922077922077922 |
| de_ancient | TopofMid | 27 | 28 | 31 | 0.889 | 0.250 | 0.548 | -0.2983870967741935 |
| de_nuke | Observation | 51 | 23 | 42 | 0.980 | 0.261 | 0.571 | -0.3105590062111801 |
| de_mirage | Connector | 183 | 134 | 150 | 0.978 | 0.351 | 0.667 | -0.3159203980099502 |
| de_ancient | SideEntrance | 307 | 185 | 226 | 0.961 | 0.378 | 0.695 | -0.31631188710834723 |
| de_nuke | Heaven | 44 | 28 | 14 | 1.000 | 0.250 | 0.571 | -0.3214285714285714 |
| de_anubis | Connector | 162 | 108 | 136 | 0.914 | 0.269 | 0.596 | -0.3270697167755991 |
| de_dust2 | Hole | 41 | 32 | 31 | 1.000 | 0.219 | 0.548 | -0.3296370967741935 |