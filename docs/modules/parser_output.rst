Example Parser Output
===================

When you use `.parse()` on the `Demo` file, you populate a series of properties on the `Demo` class. The following are available properties after parsing:

- :ref:`header`
- :ref:`rounds`
- :ref:`bomb`
- :ref:`kills`
- :ref:`damages`
- :ref:`shots`
- :ref:`grenades`
- :ref:`infernos`
- :ref:`smokes`
- :ref:`footsteps`
- :ref:`ticks`

Their contents depend on the ``player_props`` and ``other_props`` that are passed in during parsing. Keep in mind that these are Polars (``import polars as pl``) dataframes. To convert them to Pandas, you can do `df.to_pandas()`.

header
------
The `.header` property contains general information about how the demo was recorded. Notably, it contains the map information.

.. code-block:: python

    {
        'allow_clientside_particles': True, 
        'demo_version_name': 'valve_demo_2', 
        'client_name': 'SourceTV Demo', 
        'allow_clientside_entities': True, 
        'map_name': 'de_mirage', 
        'demo_version_guid': '8e9d71ab-04a1-4c01-bb61-acfede27c046', 
        'fullpackets_version': '2', 
        'server_name': 'BLAST Bounty CS2 Server', 
        'game_directory': '/home/dathost/cs2_linux/game/csgo', 
        'demo_file_stamp': 'PBDEMS2\x00', 
        'addons': '', 
        'network_protocol': '14059'
    }


rounds
------
The `.rounds` property contains information on when the important round phase change events (start/freeze/bomb plant/end/official end) occur.

.. code-block:: none

    ┌───────────┬────────┬────────────┬────────┬───┬────────┬───────────────┬────────────┬─────────────┐
    │ round_num ┆ start  ┆ freeze_end ┆ end    ┆ … ┆ winner ┆ reason        ┆ bomb_plant ┆ bomb_site   │
    │ ---       ┆ ---    ┆ ---        ┆ ---    ┆   ┆ ---    ┆ ---           ┆ ---        ┆ ---         │
    │ u32       ┆ i32    ┆ i32        ┆ i32    ┆   ┆ str    ┆ str           ┆ i64        ┆ str         │
    ╞═══════════╪════════╪════════════╪════════╪═══╪════════╪═══════════════╪════════════╪═════════════╡
    │ 1         ┆ 209    ┆ 4806       ┆ 7211   ┆ … ┆ CT     ┆ t_killed      ┆ null       ┆ not_planted │
    │ 2         ┆ 7659   ┆ 8939       ┆ 13602  ┆ … ┆ CT     ┆ t_killed      ┆ null       ┆ not_planted │
    │ 3         ┆ 14050  ┆ 15330      ┆ 23461  ┆ … ┆ T      ┆ bomb_exploded ┆ 20837      ┆ bombsite_b  │
    │ 4         ┆ 23909  ┆ 25189      ┆ 32702  ┆ … ┆ T      ┆ bomb_exploded ┆ 30078      ┆ bombsite_b  │
    │ 5         ┆ 33150  ┆ 37102      ┆ 40372  ┆ … ┆ T      ┆ ct_killed     ┆ 39741      ┆ bombsite_b  │
    │ …         ┆ …      ┆ …          ┆ …      ┆ … ┆ …      ┆ …             ┆ …          ┆ …           │
    │ 18        ┆ 144112 ┆ 148122     ┆ 155529 ┆ … ┆ T      ┆ ct_killed     ┆ 154014     ┆ bombsite_b  │
    │ 19        ┆ 155977 ┆ 157257     ┆ 163644 ┆ … ┆ CT     ┆ bomb_defused  ┆ 162290     ┆ bombsite_b  │
    │ 20        ┆ 164092 ┆ 165372     ┆ 173711 ┆ … ┆ T      ┆ bomb_exploded ┆ 171087     ┆ bombsite_b  │
    │ 21        ┆ 174159 ┆ 178124     ┆ 185501 ┆ … ┆ CT     ┆ bomb_defused  ┆ 184616     ┆ bombsite_b  │
    │ 22        ┆ 185949 ┆ 187229     ┆ 193492 ┆ … ┆ CT     ┆ bomb_defused  ┆ 191852     ┆ bombsite_b  │
    └───────────┴────────┴────────────┴────────┴───┴────────┴───────────────┴────────────┴─────────────┘


bomb
----
The `.bomb` property contains information on bomb pickup, drop, plant, defuse and detonate events.

.. code-block:: none

    ┌────────┬────────┬──────────────┬──────────────┬─────────────┬───────────────────┬────────┬───────────┐
    │ tick   ┆ event  ┆ X            ┆ Y            ┆ Z           ┆ steamid           ┆ name   ┆ bombsite  │
    │ ---    ┆ ---    ┆ ---          ┆ ---          ┆ ---         ┆ ---               ┆ ---    ┆ ---       │
    │ i32    ┆ str    ┆ f32          ┆ f32          ┆ f32         ┆ str               ┆ str    ┆ str       │
    ╞════════╪════════╪══════════════╪══════════════╪═════════════╪═══════════════════╪════════╪═══════════╡
    │ 4856   ┆ pickup ┆ 1227.36499   ┆ -162.933472  ┆ -165.088745 ┆ 76561198074762801 ┆ m0NESY ┆ null      │
    │ 4969   ┆ drop   ┆ 1134.897705  ┆ 264.215302   ┆ -110.578125 ┆ 76561198074762801 ┆ m0NESY ┆ null      │
    │ 5010   ┆ pickup ┆ 1021.556702  ┆ 558.786499   ┆ -261.331665 ┆ 76561197982141573 ┆ Snax   ┆ null      │
    │ 7014   ┆ drop   ┆ -1893.958374 ┆ 569.304993   ┆ -167.96875  ┆ 76561197982141573 ┆ Snax   ┆ null      │
    │ 7659   ┆ pickup ┆ -1893.958374 ┆ 569.304993   ┆ -167.96875  ┆ 76561197982141573 ┆ Snax   ┆ null      │
    │ …      ┆ …      ┆ …            ┆ …            ┆ …           ┆ …                 ┆ …      ┆ …         │
    │ 185501 ┆ defuse ┆ -2011.643555 ┆ 381.830597   ┆ -159.96875  ┆ 76561198012872053 ┆ huNter ┆ BombsiteB │
    │ 185949 ┆ pickup ┆ -1987.032227 ┆ 438.601532   ┆ -159.96875  ┆ 76561197961491680 ┆ tabseN ┆ null      │
    │ 187297 ┆ drop   ┆ 1267.311768  ┆ -559.58905   ┆ -163.96875  ┆ 76561197961491680 ┆ tabseN ┆ null      │
    │ 187964 ┆ pickup ┆ 1104.031372  ┆ -890.6875    ┆ -261.109253 ┆ 76561198139604328 ┆ hyped  ┆ null      │
    │ 191852 ┆ plant  ┆ -602.035461  ┆ -2155.072021 ┆ -179.96875  ┆ 76561198139604328 ┆ hyped  ┆ BombsiteA │
    └────────┴────────┴──────────────┴──────────────┴─────────────┴───────────────────┴────────┴───────────┘


kills
-----
The `.kills` property contains information on when a player kills another player.

.. code-block:: none

    ┌───────────────┬──────────────┬──────────────┬─────────────┬───┬──────────────────────┬───────────────┬───────────────────────────┬──────┐
    │ assistedflash ┆ assister_X   ┆ assister_Y   ┆ assister_Z  ┆ … ┆ weapon_fauxitemid    ┆ weapon_itemid ┆ weapon_originalowner_xuid ┆ wipe │
    │ ---           ┆ ---          ┆ ---          ┆ ---         ┆   ┆ ---                  ┆ ---           ┆ ---                       ┆ ---  │
    │ bool          ┆ f32          ┆ f32          ┆ f32         ┆   ┆ str                  ┆ str           ┆ str                       ┆ i32  │
    ╞═══════════════╪══════════════╪══════════════╪═════════════╪═══╪══════════════════════╪═══════════════╪═══════════════════════════╪══════╡
    │ false         ┆ null         ┆ null         ┆ null        ┆ … ┆ 17293822569135734845 ┆ 41416174006   ┆                           ┆ 0    │
    │ false         ┆ null         ┆ null         ┆ null        ┆ … ┆ 17293822569135734845 ┆ 41416174006   ┆                           ┆ 0    │
    │ false         ┆ -1291.362671 ┆ 245.081375   ┆ -167.404968 ┆ … ┆ 17293822569123217469 ┆ 41460454457   ┆                           ┆ 0    │
    │ false         ┆ null         ┆ null         ┆ null        ┆ … ┆ 17293822569135734845 ┆ 16032582195   ┆                           ┆ 0    │
    │ false         ┆ null         ┆ null         ┆ null        ┆ … ┆ 17293822569177153597 ┆ 27259257760   ┆                           ┆ 0    │
    │ …             ┆ …            ┆ …            ┆ …           ┆ … ┆ …                    ┆ …             ┆ …                         ┆ …    │
    │ false         ┆ -1040.831421 ┆ -2322.427246 ┆ -167.96875  ┆ … ┆ 17293822569144582151 ┆ 41401525409   ┆                           ┆ 0    │
    │ false         ┆ -1040.831421 ┆ -2322.427246 ┆ -167.96875  ┆ … ┆ 17293822569144582151 ┆ 41401525409   ┆                           ┆ 0    │
    │ false         ┆ null         ┆ null         ┆ null        ┆ … ┆ 17293822569120989193 ┆ 40012573917   ┆                           ┆ 0    │
    │ false         ┆ null         ┆ null         ┆ null        ┆ … ┆ 17293822569168306236 ┆ 40738411466   ┆                           ┆ 0    │
    │ false         ┆ null         ┆ null         ┆ null        ┆ … ┆                      ┆               ┆                           ┆ 0    │
    └───────────────┴──────────────┴──────────────┴─────────────┴───┴──────────────────────┴───────────────┴───────────────────────────┴──────┘


damages
-------
The `.damages` property contains information on when a player damages another player.

.. code-block:: none

    ┌───────┬──────────────┬──────────────┬─────────────┬───┬─────────────┬───────────────────┬─────────┬─────────────────┐
    │ armor ┆ attacker_X   ┆ attacker_Y   ┆ attacker_Z  ┆ … ┆ victim_name ┆ victim_steamid    ┆ weapon  ┆ dmg_health_real │
    │ ---   ┆ ---          ┆ ---          ┆ ---         ┆   ┆ ---         ┆ ---               ┆ ---     ┆ ---             │
    │ i32   ┆ f32          ┆ f32          ┆ f32         ┆   ┆ str         ┆ str               ┆ str     ┆ i32             │
    ╞═══════╪══════════════╪══════════════╪═════════════╪═══╪═════════════╪═══════════════════╪═════════╪═════════════════╡
    │ 93    ┆ -1375.963867 ┆ 322.930206   ┆ -167.96875  ┆ … ┆ m0NESY      ┆ 76561198074762801 ┆ hkp2000 ┆ 16              │
    │ 99    ┆ -1375.970947 ┆ 314.907135   ┆ -167.96875  ┆ … ┆ malbsMd     ┆ 76561198080703143 ┆ hkp2000 ┆ 2               │
    │ 93    ┆ -1293.294067 ┆ 267.573212   ┆ -167.469788 ┆ … ┆ m0NESY      ┆ 76561198074762801 ┆ hkp2000 ┆ 84              │
    │ 99    ┆ -829.127258  ┆ 42.196011    ┆ -167.040222 ┆ … ┆ malbsMd     ┆ 76561198080703143 ┆ hkp2000 ┆ 0               │
    │ 92    ┆ -1448.172852 ┆ 129.08699    ┆ -166.96875  ┆ … ┆ malbsMd     ┆ 76561198080703143 ┆ hkp2000 ┆ 15              │
    │ …     ┆ …            ┆ …            ┆ …           ┆ … ┆ …           ┆ …                 ┆ …       ┆ …               │
    │ 74    ┆ -1079.625488 ┆ -1461.253052 ┆ -164.593445 ┆ … ┆ kyuubii     ┆ 76561198144926364 ┆ ak47    ┆ 33              │
    │ 70    ┆ -1080.968262 ┆ -1465.090576 ┆ -164.466309 ┆ … ┆ kyuubii     ┆ 76561198144926364 ┆ ak47    ┆ 33              │
    │ 99    ┆ -1541.816895 ┆ -2364.043701 ┆ -244.644287 ┆ … ┆ Snax        ┆ 76561197982141573 ┆ awp     ┆ 100             │
    │ 77    ┆ -758.870544  ┆ -1669.898682 ┆ -172.558472 ┆ … ┆ hyped       ┆ 76561198139604328 ┆ m4a1    ┆ 82              │
    │ 72    ┆ -758.761292  ┆ -1670.187744 ┆ -172.587402 ┆ … ┆ hyped       ┆ 76561198139604328 ┆ m4a1    ┆ 9               │
    └───────┴──────────────┴──────────────┴─────────────┴───┴─────────────┴───────────────────┴─────────┴─────────────────┘


shots
-----
The `.shots` property contains information on when a player shoots their weapon.

.. code-block:: none

    ┌──────────┬────────┬──────────────┬──────────────┬───┬────────────────────────┬─────────────┬───────────────────┬─────────────────────────┐
    │ silenced ┆ tick   ┆ player_X     ┆ player_Y     ┆ … ┆ player_last_place_name ┆ player_name ┆ player_steamid    ┆ weapon                  │
    │ ---      ┆ ---    ┆ ---          ┆ ---          ┆   ┆ ---                    ┆ ---         ┆ ---               ┆ ---                     │
    │ bool     ┆ i32    ┆ f32          ┆ f32          ┆   ┆ str                    ┆ str         ┆ str               ┆ str                     │
    ╞══════════╪════════╪══════════════╪══════════════╪═══╪════════════════════════╪═════════════╪═══════════════════╪═════════════════════════╡
    │ false    ┆ 5090   ┆ -1379.150391 ┆ -973.78064   ┆ … ┆ CTSpawn                ┆ JDC         ┆ 76561198078771373 ┆ weapon_knife_m9_bayonet │
    │ false    ┆ 5132   ┆ -1323.212036 ┆ -967.742065  ┆ … ┆ CTSpawn                ┆ JDC         ┆ 76561198078771373 ┆ weapon_knife_m9_bayonet │
    │ false    ┆ 5231   ┆ -1914.81958  ┆ -323.105957  ┆ … ┆ Shop                   ┆ hyped       ┆ 76561198139604328 ┆ weapon_knife_butterfly  │
    │ false    ┆ 6240   ┆ -161.028076  ┆ 576.19574    ┆ … ┆ BackAlley              ┆ HeavyGod    ┆ 76561198068002993 ┆ weapon_smokegrenade     │
    │ false    ┆ 6266   ┆ -160.031372  ┆ 887.971619   ┆ … ┆ BackAlley              ┆ Snax        ┆ 76561197982141573 ┆ weapon_smokegrenade     │
    │ …        ┆ …      ┆ …            ┆ …            ┆ … ┆ …                      ┆ …           ┆ …                 ┆ …                       │
    │ false    ┆ 191981 ┆ -1242.619263 ┆ -1435.148438 ┆ … ┆ Jungle                 ┆ malbsMd     ┆ 76561198080703143 ┆ weapon_flashbang        │
    │ true     ┆ 192969 ┆ -758.870544  ┆ -1669.898682 ┆ … ┆ BombsiteA              ┆ malbsMd     ┆ 76561198080703143 ┆ weapon_m4a1_silencer    │
    │ true     ┆ 192975 ┆ -758.761292  ┆ -1670.187744 ┆ … ┆ BombsiteA              ┆ malbsMd     ┆ 76561198080703143 ┆ weapon_m4a1_silencer    │
    │ true     ┆ 192982 ┆ -758.760498  ┆ -1669.444092 ┆ … ┆ BombsiteA              ┆ malbsMd     ┆ 76561198080703143 ┆ weapon_m4a1_silencer    │
    │ true     ┆ 192988 ┆ -758.760498  ┆ -1669.444092 ┆ … ┆ BombsiteA              ┆ malbsMd     ┆ 76561198080703143 ┆ weapon_m4a1_silencer    │
    └──────────┴────────┴──────────────┴──────────────┴───┴────────────────────────┴─────────────┴───────────────────┴─────────────────────────┘


grenades
--------
The `.grenades` property contains information on when a player throws a grenade.

.. code-block:: none

    ┌───────────────────┬──────────┬──────────────┬────────┬───┬─────────────┬────────────┬───────────┬───────────┐
    │ thrower_steamid   ┆ thrower  ┆ grenade_type ┆ tick   ┆ … ┆ Y           ┆ Z          ┆ entity_id ┆ round_num │
    │ ---               ┆ ---      ┆ ---          ┆ ---    ┆   ┆ ---         ┆ ---        ┆ ---       ┆ ---       │
    │ u64               ┆ str      ┆ str          ┆ i32    ┆   ┆ f32         ┆ f32        ┆ i32       ┆ u32       │
    ╞═══════════════════╪══════════╪══════════════╪════════╪═══╪═════════════╪════════════╪═══════════╪═══════════╡
    │ 76561198068002993 ┆ HeavyGod ┆ smoke        ┆ 6253   ┆ … ┆ 565.6875    ┆ 22.40625   ┆ 111       ┆ 1         │
    │ 76561198068002993 ┆ HeavyGod ┆ smoke        ┆ 6254   ┆ … ┆ 562.53125   ┆ 28.5       ┆ 111       ┆ 1         │
    │ 76561198068002993 ┆ HeavyGod ┆ smoke        ┆ 6255   ┆ … ┆ 559.375     ┆ 34.5       ┆ 111       ┆ 1         │
    │ 76561198068002993 ┆ HeavyGod ┆ smoke        ┆ 6256   ┆ … ┆ 556.21875   ┆ 40.437496  ┆ 111       ┆ 1         │
    │ 76561198068002993 ┆ HeavyGod ┆ smoke        ┆ 6257   ┆ … ┆ 553.0625    ┆ 46.28125   ┆ 111       ┆ 1         │
    │ …                 ┆ …        ┆ …            ┆ …      ┆ … ┆ …           ┆ …          ┆ …         ┆ …         │
    │ 76561198080703143 ┆ malbsMd  ┆ flashbang    ┆ 192087 ┆ … ┆ -1589.6875  ┆ -148.40625 ┆ 249       ┆ 22        │
    │ 76561198080703143 ┆ malbsMd  ┆ flashbang    ┆ 192088 ┆ … ┆ -1591.1875  ┆ -152.875   ┆ 249       ┆ 22        │
    │ 76561198080703143 ┆ malbsMd  ┆ flashbang    ┆ 192089 ┆ … ┆ -1592.6875  ┆ -157.40625 ┆ 249       ┆ 22        │
    │ 76561198080703143 ┆ malbsMd  ┆ flashbang    ┆ 192090 ┆ … ┆ -1594.21875 ┆ -162.0625  ┆ 249       ┆ 22        │
    │ 76561198080703143 ┆ malbsMd  ┆ flashbang    ┆ 192091 ┆ … ┆ -1595.59375 ┆ -165.625   ┆ 249       ┆ 22        │
    └───────────────────┴──────────┴──────────────┴────────┴───┴─────────────┴────────────┴───────────┴───────────┘


infernos
--------
The `.infernos` property contains information on when molotov or incendiary grenade starts and expires.

.. code-block:: none

    ┌───────────┬────────────┬──────────┬──────────────┬───┬───────────────────┬──────────────┬──────────────┬────────────┐
    │ entity_id ┆ start_tick ┆ end_tick ┆ thrower_X    ┆ … ┆ thrower_steamid   ┆ X            ┆ Y            ┆ Z          │
    │ ---       ┆ ---        ┆ ---      ┆ ---          ┆   ┆ ---               ┆ ---          ┆ ---          ┆ ---        │
    │ i64       ┆ i64        ┆ i64      ┆ f64          ┆   ┆ str               ┆ f64          ┆ f64          ┆ f64        │
    ╞═══════════╪════════════╪══════════╪══════════════╪═══╪═══════════════════╪══════════════╪══════════════╪════════════╡
    │ 438       ┆ 9479       ┆ 9832     ┆ -749.3255    ┆ … ┆ 76561198144926364 ┆ 249.482285   ┆ -1527.261108 ┆ -173.96875 │
    │ 46        ┆ 15699      ┆ 15931    ┆ -762.805664  ┆ … ┆ 76561198144926364 ┆ 227.161819   ┆ -1515.966675 ┆ -173.96875 │
    │ 218       ┆ 15990      ┆ 16343    ┆ -1563.132568 ┆ … ┆ 76561198170631091 ┆ -1536.188232 ┆ 712.675232   ┆ -45.96875  │
    │ 43        ┆ 16131      ┆ 16228    ┆ -1258.452271 ┆ … ┆ 76561198139604328 ┆ 451.550842   ┆ -629.696655  ┆ -159.96875 │
    │ 202       ┆ 18718      ┆ 19071    ┆ -786.658203  ┆ … ┆ 76561197961491680 ┆ -1007.844727 ┆ -601.723022  ┆ -285.96875 │
    │ …         ┆ …          ┆ …        ┆ …            ┆ … ┆ …                 ┆ …            ┆ …            ┆ …          │
    │ 136       ┆ 188071     ┆ 188425   ┆ -1964.792358 ┆ … ┆ 76561197982141573 ┆ -1220.075073 ┆ 687.893616   ┆ -77.96875  │
    │ 171       ┆ 188370     ┆ 188724   ┆ -611.331116  ┆ … ┆ 76561198080703143 ┆ -951.713867  ┆ -276.406525  ┆ -361.96875 │
    │ 217       ┆ 189287     ┆ 189736   ┆ 26.230442    ┆ … ┆ 76561197961491680 ┆ 94.700676    ┆ -2335.717041 ┆ -37.96875  │
    │ 38        ┆ 190009     ┆ 190458   ┆ -388.368042  ┆ … ┆ 76561198139604328 ┆ -1190.271606 ┆ -1302.557983 ┆ -168.82959 │
    │ 240       ┆ 191123     ┆ 191477   ┆ -554.03479   ┆ … ┆ 76561198012872053 ┆ -642.059753  ┆ -2190.879639 ┆ -180.0     │
    └───────────┴────────────┴──────────┴──────────────┴───┴───────────────────┴──────────────┴──────────────┴────────────┘


smokes
------
The `.smokes` property contains information on when a smoke grenade starts and expires.

.. code-block:: none

    ┌───────────┬────────────┬──────────┬──────────────┬───┬───────────────────┬──────────────┬──────────────┬─────────────┐
    │ entity_id ┆ start_tick ┆ end_tick ┆ thrower_X    ┆ … ┆ thrower_steamid   ┆ X            ┆ Y            ┆ Z           │
    │ ---       ┆ ---        ┆ ---      ┆ ---          ┆   ┆ ---               ┆ ---          ┆ ---          ┆ ---         │
    │ i64       ┆ i64        ┆ i64      ┆ f64          ┆   ┆ str               ┆ f64          ┆ f64          ┆ f64         │
    ╞═══════════╪════════════╪══════════╪══════════════╪═══╪═══════════════════╪══════════════╪══════════════╪═════════════╡
    │ 111       ┆ 6585       ┆ null     ┆ -603.621338  ┆ … ┆ 76561198068002993 ┆ -1273.158081 ┆ 199.570801   ┆ -166.040344 │
    │ 193       ┆ 6753       ┆ null     ┆ -1417.788574 ┆ … ┆ 76561197982141573 ┆ -1979.307251 ┆ -325.090729  ┆ -165.96875  │
    │ 439       ┆ 9882       ┆ 11294    ┆ -1853.035889 ┆ … ┆ 76561198170631091 ┆ -1437.317139 ┆ 725.867432   ┆ -53.96875   │
    │ 450       ┆ 11309      ┆ 12721    ┆ -828.063354  ┆ … ┆ 76561197961491680 ┆ -617.765198  ┆ -875.580383  ┆ -253.96875  │
    │ 465       ┆ 12596      ┆ 14008    ┆ -578.492371  ┆ … ┆ 76561198078771373 ┆ -634.322815  ┆ -745.603333  ┆ -264.626923 │
    │ …         ┆ …          ┆ …        ┆ …            ┆ … ┆ …                 ┆ …            ┆ …            ┆ …           │
    │ 36        ┆ 189136     ┆ 190548   ┆ 773.494446   ┆ … ┆ 76561198144926364 ┆ -512.126648  ┆ -1611.015625 ┆ -37.971172  │
    │ 195       ┆ 189220     ┆ 190632   ┆ -805.207947  ┆ … ┆ 76561198080703143 ┆ -633.220032  ┆ -726.947754  ┆ -266.27713  │
    │ 173       ┆ 189249     ┆ 190661   ┆ -1026.025146 ┆ … ┆ 76561198074762801 ┆ 254.503632   ┆ -1530.493286 ┆ -173.96875  │
    │ 411       ┆ 189259     ┆ 190671   ┆ 530.297791   ┆ … ┆ 76561198144926364 ┆ -808.669983  ┆ -1622.820679 ┆ 18.03125    │
    │ 101       ┆ 190612     ┆ 192024   ┆ -1063.140381 ┆ … ┆ 76561198012872053 ┆ -684.59198   ┆ -1636.631226 ┆ -169.96875  │
    └───────────┴────────────┴──────────┴──────────────┴───┴───────────────────┴──────────────┴──────────────┴─────────────┘


footsteps
---------
The `.footsteps` property contains information on when a player makes a footstep.

.. code-block:: none

    ┌──────────┬────────┬───────┬────────┬───┬───────────────┬────────────────────────┬─────────────┬───────────────────┐
    │ duration ┆ radius ┆ step  ┆ tick   ┆ … ┆ player_health ┆ player_last_place_name ┆ player_name ┆ player_steamid    │
    │ ---      ┆ ---    ┆ ---   ┆ ---    ┆   ┆ ---           ┆ ---                    ┆ ---         ┆ ---               │
    │ f32      ┆ i32    ┆ bool  ┆ i32    ┆   ┆ i32           ┆ str                    ┆ str         ┆ str               │
    ╞══════════╪════════╪═══════╪════════╪═══╪═══════════════╪════════════════════════╪═════════════╪═══════════════════╡
    │ 0.1      ┆ 1100   ┆ false ┆ 209    ┆ … ┆ 100           ┆ CTSpawn                ┆ tabseN      ┆ 76561197961491680 │
    │ 0.1      ┆ 1100   ┆ false ┆ 209    ┆ … ┆ 100           ┆ CTSpawn                ┆ tabseN      ┆ 76561197961491680 │
    │ 0.1      ┆ 1100   ┆ false ┆ 209    ┆ … ┆ 100           ┆ TSpawn                 ┆ Snax        ┆ 76561197982141573 │
    │ 0.1      ┆ 1100   ┆ false ┆ 209    ┆ … ┆ 100           ┆ TSpawn                 ┆ Snax        ┆ 76561197982141573 │
    │ 0.1      ┆ 1100   ┆ false ┆ 209    ┆ … ┆ 100           ┆ TSpawn                 ┆ huNter      ┆ 76561198012872053 │
    │ …        ┆ …      ┆ …     ┆ …      ┆ … ┆ …             ┆ …                      ┆ …           ┆ …                 │
    │ 0.5      ┆ 1100   ┆ true  ┆ 193250 ┆ … ┆ 100           ┆ BombsiteA              ┆ huNter      ┆ 76561198012872053 │
    │ 0.1      ┆ 1400   ┆ false ┆ 193630 ┆ … ┆ 100           ┆ BombsiteA              ┆ huNter      ┆ 76561198012872053 │
    │ 0.1      ┆ 1070   ┆ false ┆ 193630 ┆ … ┆ 100           ┆ BombsiteA              ┆ huNter      ┆ 76561198012872053 │
    │ 0.1      ┆ 1070   ┆ false ┆ 193630 ┆ … ┆ 100           ┆ BombsiteA              ┆ huNter      ┆ 76561198012872053 │
    │ 0.1      ┆ 1070   ┆ false ┆ 193630 ┆ … ┆ 100           ┆ BombsiteA              ┆ huNter      ┆ 76561198012872053 │
    └──────────┴────────┴───────┴────────┴───┴───────────────┴────────────────────────┴─────────────┴───────────────────┘


ticks
-----
The `.ticks` property contains information per player per tick. Ticks during timeouts, warmup, etc. are excluded.

.. code-block:: none

    ┌────────┬───────────────────┬──────────┬───────────┐
    │ tick   ┆ steamid           ┆ name     ┆ round_num │
    │ ---    ┆ ---               ┆ ---      ┆ ---       │
    │ i32    ┆ u64               ┆ str      ┆ u32       │
    ╞════════╪═══════════════════╪══════════╪═══════════╡
    │ 210    ┆ 76561198068002993 ┆ HeavyGod ┆ 1         │
    │ 210    ┆ 76561198144926364 ┆ kyuubii  ┆ 1         │
    │ 210    ┆ 76561198078771373 ┆ JDC      ┆ 1         │
    │ 210    ┆ 76561198080703143 ┆ malbsMd  ┆ 1         │
    │ 210    ┆ 76561198170631091 ┆ Krimbo   ┆ 1         │
    │ …      ┆ …                 ┆ …        ┆ …         │
    │ 193491 ┆ 76561198012872053 ┆ huNter   ┆ 22        │
    │ 193491 ┆ 76561197961491680 ┆ tabseN   ┆ 22        │
    │ 193491 ┆ 76561198074762801 ┆ m0NESY   ┆ 22        │
    │ 193491 ┆ 76561198139604328 ┆ hyped    ┆ 22        │
    │ 193491 ┆ 76561197982141573 ┆ Snax     ┆ 22        │
    └────────┴───────────────────┴──────────┴───────────┘
