❯ python -m cellular_automaton.ca_analyze leaderboard \                                                                                                                                                                                  ─╯
--where status=completed \
--where training_pairs='1:1|2:2|3:3|4:4|5:5|6:6' \
--role repeat_horizon_diagnostic \
--select-pair 9:9 \
--report-pairs 7:7 8:8 9:9 10:10 11:11 12:12 \
--metric cell_accuracy
rank  run                     model           protocol  seed  data  architecture                             wd   clip  best 9:9 step  7:7       8:8       9:9       10:10     11:11     12:12     config id 
----  ----------------------  --------------  --------  ----  ----  ---------------------------------------  ---  ----  -------------  --------  --------  --------  --------  --------  --------  ----------
1     run_62__but_full_depth  but_full_depth  P1        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   7250           1.000000  1.000000  0.999992  0.999802  0.998890  0.996487  eccc76dd4b
2     run_63__but_full_depth  but_full_depth  P1        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   5750           0.999989  0.999977  0.999985  0.999928  0.999748  0.999451  a6c186daca
3     run_61__but_full_depth  but_full_depth  P1        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   6500           0.999989  0.999947  0.999744  0.998970  0.992500  0.946739  9f78ec1ff6
4     run_58__ca_cotf         ca_cotf         P1        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   3750           0.999733  0.998913  0.992546  0.947563  0.661900  0.507732  1e96cf3382
5     run_59__ca_cotf         ca_cotf         P1        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   7500           0.999817  0.996464  0.934128  0.566551  0.500584  0.502281  bd284c8b35
6     run_60__ca_cotf         ca_cotf         P1        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   6000           0.998146  0.983807  0.755188  0.512238  0.492012  0.498116  ce6f78e24e




this pattern seems to be repeating. BUT extrapolates way better even when CoTFormer is given more training budget (but not necessarily more examples):


❯ python -m cellular_automaton.ca_analyze leaderboard \                                                                                                                                                                                  ─╯
  --where training_pairs='1:1|2:2|3:3|4:4' \
  --role repeat_horizon_diagnostic \
  --select-pair 7:7 \
  --report-pairs 5:5 6:6 7:7 8:8 9:9 10:10 \
  --metric cell_accuracy
rank  run                     model           protocol  seed  data  architecture                             wd    clip  best 7:7 step  5:5       6:6       7:7       8:8       9:9       10:10     config id 
----  ----------------------  --------------  --------  ----  ----  ---------------------------------------  ----  ----  -------------  --------  --------  --------  --------  --------  --------  ----------
1     run_12__but_full_depth  but_full_depth  P1        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.15  0.9   5000           1.000000  0.999996  0.999931  0.999397  0.997299  0.992859  b4803cdef2
2     run_24__but_full_depth  but_full_depth  P2        3     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   2750           0.999996  0.999989  0.999920  0.999786  0.999336  0.998409  e81f5c2e68
3     run_23__but_full_depth  but_full_depth  P2        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   3750           1.000000  0.999977  0.999672  0.998672  0.996159  0.990818  6fb4348fc2
4     run_25__but_full_depth  but_full_depth  P2        4     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   5000           0.999969  0.999851  0.998917  0.989700  0.966228  0.929367  c4544ada87
5     run_2__but_full_depth   but_full_depth  P3        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   4250           1.000000  0.999889  0.998158  0.989567  0.968105  —         98a2691905
6     run_7__but_full_depth   but_full_depth  P1        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   4250           1.000000  0.999889  0.998158  0.989567  0.968105  0.919369  a5099d151b
7     run_22__but_full_depth  but_full_depth  P2        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   3750           0.999222  0.995750  0.980919  0.935799  0.838924  0.676956  0665ddae10
8     run_36__ca_cotf         ca_cotf         P2        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   8500           0.999256  0.993855  0.877007  0.555126  0.501633  0.501827  f1546aae12
9     run_9__ca_cotf          ca_cotf         P1        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   4750           0.999344  0.962833  0.667236  0.506031  0.500011  0.502018  87f8b01510
10    run_52__ca_cotf         ca_cotf         P2        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.0   5000           0.997166  0.940548  0.632553  0.514050  0.501690  0.501781  7d497238c9
11    run_54__ca_cotf         ca_cotf         P2        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.0   5000           0.994881  0.955505  0.612259  0.492298  0.499527  0.501404  d11bd59571
12    run_13__ca_cotf         ca_cotf         P1        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.15  0.9   4500           0.995335  0.938023  0.609337  0.518688  0.500603  0.500320  e376289aa6
13    run_11__ca_cotf         ca_cotf         P1        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   1.0   4500           0.971123  0.854828  0.605789  0.504799  0.496037  0.497700  513574d7fd
14    run_48__ca_cotf         ca_cotf         P2        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.05  0.9   4250           0.954567  0.805138  0.562080  0.498722  0.499241  0.500534  62e99e44d2
15    run_56__ca_cotf         ca_cotf         P2        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.0   3750           0.898029  0.683620  0.523174  0.497860  0.499138  0.499264  666bc92714
16    run_51__ca_cotf         ca_cotf         P2        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.2   0.9   4500           0.967461  0.709061  0.519371  0.498940  0.499508  0.500477  558a01420d
17    run_35__ca_cotf         ca_cotf         P2        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   8750           0.869728  0.588116  0.512863  0.499092  0.500118  0.500919  314d166ac9
18    run_15__but_full_depth  but_full_depth  P1        2     1     bidirectional/rotary d64h4 layers=1/1/1  0.15  0.9   2000           0.535595  0.489697  0.510685  0.503551  0.498363  0.499874  dab8f91495
19    run_8__ca_cotf          ca_cotf         P1        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   750            0.493675  0.496719  0.508793  0.497555  0.499279  0.501530  ac1e8d646c
20    run_37__ca_cotf         ca_cotf         P2        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.0   0.9   1250           0.502018  0.487068  0.507202  0.499676  0.499424  0.501431  bec3cfd6e6
21    run_42__ca_cotf         ca_cotf         P2        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.0   0.9   2750           0.505798  0.500389  0.506950  0.501011  0.498749  0.499966  f9dcc684c4
22    run_55__ca_cotf         ca_cotf         P2        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.5   2000           0.508461  0.497505  0.505043  0.497639  0.499935  0.499214  2332bd1223
23    run_38__ca_cotf         ca_cotf         P2        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.05  0.9   1000           0.497791  0.498505  0.504772  0.497871  0.500725  0.500305  a2780459ee
24    run_41__ca_cotf         ca_cotf         P2        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.2   0.9   4750           0.535549  0.500977  0.504055  0.501469  0.499809  0.499279  1bf0f57464
25    run_40__ca_cotf         ca_cotf         P2        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.15  0.9   3750           0.495922  0.500099  0.503735  0.498329  0.499802  0.501675  b5d8273c99
26    run_43__ca_cotf         ca_cotf         P2        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.05  0.9   4750           0.523361  0.499443  0.503723  0.500328  0.499832  0.498856  cf04f572c0
27    run_47__ca_cotf         ca_cotf         P2        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.0   0.9   4750           0.846237  0.532188  0.503601  0.498936  0.502430  0.498158  20e617c253
28    run_57__ca_cotf         ca_cotf         P2        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.5   250            0.499077  0.498505  0.503429  0.499344  0.499786  0.498920  d2f28dd7a3
29    run_34__ca_cotf         ca_cotf         P2        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   5750           0.498863  0.500992  0.502316  0.499462  0.500683  0.500698  c188e3f301
30    run_53__ca_cotf         ca_cotf         P2        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.5   2500           0.495144  0.500160  0.501839  0.501671  0.501461  0.500492  f5ec845051
31    run_19__ca_cotf         ca_cotf         P2        3     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   3000           0.514050  0.495903  0.501507  0.499680  0.499969  0.499428  718b060803
32    run_14__ca_cotf         ca_cotf         P1        2     1     bidirectional/rotary d64h4 layers=1/1/1  0.15  0.9   2250           0.520947  0.494766  0.501335  0.500126  0.500412  0.498089  1eb555b0dd
33    run_46__ca_cotf         ca_cotf         P2        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.2   0.9   3250           0.496277  0.496647  0.501331  0.500427  0.499603  0.501488  844d7e30c7
34    run_20__ca_cotf         ca_cotf         P2        4     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   3500           0.501492  0.498013  0.501328  0.499626  0.500294  0.499821  87bee60cf3
35    run_45__ca_cotf         ca_cotf         P2        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.15  0.9   2250           0.494743  0.500721  0.501236  0.501110  0.498409  0.500042  800eb3d985
36    run_6__ca_cotf          ca_cotf         P1        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1   0.9   4500           0.498375  0.497028  0.500427  0.500622  0.497746  0.499157  508f341a83

Ranked by max validation cell_accuracy on 7:7; every pair in a row is reported at that run's selected step.
WARNING: mixed evaluation protocols:
  P1 train=1:1|2:2|3:3|4:4 val=5:5|6:6 select=6:6 direct_final=8:8 diagnostic_horizons=0,1,2,3,4,5,6,7,8,9,10 repeats=1:10
  P2 train=1:1|2:2|3:3|4:4 val=5:5|6:6|7:7 select=7:7 direct_final=none diagnostic_horizons=0,1,2,3,4,5,6,7,8,9,10 repeats=1:10
  P3 train=1:1|2:2|3:3|4:4 val=5:5|6:6|7:7|8:8|9:9 select=9:9 direct_final=10:10 diagnostic_horizons=0,1,2,3,4,5,6,7,8,9 repeats=1:9
Skipped selected runs without a matching selection-pair record: run_10__ca_cotf, run_16__ca_cotf, run_17__ca_cotf, run_18__ca_cotf, run_21__but_full_depth, run_26__ca_cotf, run_27__ca_cotf, run_28__ca_cotf, run_29__ca_cotf, run_30__ca_cotf, run_31__ca_cotf, run_32__ca_cotf, run_33__ca_cotf, run_39__ca_cotf, run_44__ca_cotf, run_49__ca_cotf, run_50__ca_cotf




beyond a certain repeat count extrapolation collapses for the CoTFormer



❯   conda run -n ml python -m cellular_automaton.ca_analyze leaderboard \                                                                                                                                                                ─╯
    --where status=completed \
    --where annotations.tags~final_ceiling \
    --where training_pairs='1:1|2:2|3:3|4:4|5:5|6:6|7:7|8:8|9:9|10:10|11:11|12:12' \
    --role repeat_horizon_diagnostic \
    --select-pair 15:15 \
    --report-pairs \
      13:13 14:14 15:15 16:16 \
      17:17 18:18 19:19 20:20 \
    --metric cell_accuracy

rank  run                     model           protocol  seed  data  architecture                             wd   clip  best 15:15 step  13:13     14:14     15:15     16:16     17:17     18:18     19:19     20:20     config id 
----  ----------------------  --------------  --------  ----  ----  ---------------------------------------  ---  ----  ---------------  --------  --------  --------  --------  --------  --------  --------  --------  ----------
1     run_82__but_full_depth  but_full_depth  P1        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   13000            0.999954  0.999920  0.999920  0.999931  0.999931  0.999870  0.999847  0.999775  7de1a4d23b
2     run_79__ca_cotf         ca_cotf         P1        0     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   20000            0.999901  0.999756  0.999210  0.996784  0.975582  0.858620  0.593754  0.505241  2e5b588988
3     run_81__ca_cotf         ca_cotf         P1        2     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   24000            0.914864  0.902603  0.874599  0.762039  0.603367  0.508255  0.500381  0.499748  514364b459
4     run_80__ca_cotf         ca_cotf         P1        1     1     bidirectional/rotary d64h4 layers=0/1/0  0.1  0.9   8000             0.500088  0.499641  0.501472  0.500771  0.498569  0.500324  0.498322  0.501366  59a53e5bc0

Ranked by max validation cell_accuracy on 15:15; every pair in a row is reported at that run's selected step.
Evaluation protocol:
  P1 train=1:1|2:2|3:3|4:4|5:5|6:6|7:7|8:8|9:9|10:10|11:11|12:12 val=13:13|14:14|15:15 select=15:15 direct_final=none diagnostic_horizons=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 repeats=1:20
