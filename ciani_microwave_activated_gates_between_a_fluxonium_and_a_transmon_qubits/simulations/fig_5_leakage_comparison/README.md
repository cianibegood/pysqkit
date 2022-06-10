The results for the transmon fluxonium case were obtained by running

cr_gate_fidelity_leakage_tf_fig_5_datagen.py

using a machine with 200 cores, i.e., setting n_process = 200. 

In the code we set n_process = 1 by default, but this would lead to 
large computation times with n_points = 200. We suggest to test
the code with fewer points.

The same is valid in the transmon-transmon case using

cr_gate_leakage_tt_fig_5_datagen.py
