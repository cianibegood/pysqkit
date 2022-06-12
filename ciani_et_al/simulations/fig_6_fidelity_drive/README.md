In the simulations for the paper we included 6 levels for the fluxonium (levels_f = 6) and
3 for the transmon (levels_t = 3).

We run the simulations in parallel with n_process = 16, but in the present code we set it to 1. 
You should adapt n_process to the number of cores of your machine. In this case it is the tomography
that is executed in parallel, i.e., we obtain the evolution of each Pauli operator in parallel.

