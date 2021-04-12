"""
Jump operators for dielectric loss for a Fluxonium qubit.
"""

#%%
import numpy as np 
from pysqkit.qubits import fluxonium
from pysqkit.util import linalg
from typing import Iterable
import qutip

#%%
"""
Parameters from Schuster et al 
"""

ec_ref = 0.479 #GHz
ec = 1    # unit fixed to 1
el = 0.132/ec_ref
ej = 3.395/ec_ref 
flux = 1/2
flx = fluxonium.Fluxonium('F1', ec, el, ej, flux, dim_hilbert=50)

qdiel = 1000
beta = 100

# %%
k = 0
m = 3
flx.dielectric_jump(k, m, qdiel, beta)
jump_down, jump_up = flx.dielectric_jump(k, m, qdiel, beta, as_qobj=False)
jump_diel = flx.dielectric_loss(qdiel, beta, as_qobj=True)




# %%
