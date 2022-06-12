"""
Coupled fluxonium system.
"""

#%%
import numpy as np 
from pysqkit.qubits import fluxonium
from pysqkit.util import linalg
from typing import Iterable
import qutip

#%%
"""
Parameters for fluxonium F1 from Schuster et al
"""

ec_ref = 0.479 #GHz
ec = 1    # unit fixed to 1
el = [0.132/ec_ref, 0.15/ec_ref]
ej = [3.395/ec_ref, 3.8/ec_ref]
flux = [1/2, 1/2]
flx1 = fluxonium.Fluxonium('F1', ec, el[0], ej[0], flux[0], dim_hilbert=50)
flx2 = fluxonium.Fluxonium('F2', ec, el[1], ej[1], flux[1], dim_hilbert=50)

flx_sys = flx1.couple_to(flx2)

# %%
truncation = {'F1': 8, 'F2': 9}
flx_sys.convert_subsys_operator('F1', 'CIAO', truncated_levels=truncation)


# %%
