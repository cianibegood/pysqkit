The results for the transmon fluxonium case were obtained by running

cr_gate_fidelity_leakage_tf_fig_5_datagen.py

using a machine with 200 cores, i.e., setting n_process = 200. 
Additionally, we included 8 levels (levels_f = 8) for the fluxonium 
and 3 (levels_t = 3) for the transmon. 

The simulations also compute the fidelities in this case and as
a consequence the a single run of the function takes a considerable amount of time 
on a standard laptop. A single evaluation of the fidelity takes roughly 500 s.
In the code we set n_process = 1 by default, but this would lead to 
large computation times with n_points = 200. We suggest to test
the code with fewer points.

The same is valid in the transmon-transmon case using

cr_gate_leakage_tt_fig_5_datagen.py

In these simulations we always included 3 levels for the transmon (levels_t = 3).
