"""
Matrix elements of relevant operators for the 
fluxonium qubit
"""

#%%

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import sys
from pathlib import Path
parent_path = str(Path(__file__).resolve().parents[2])
sys.path.append(parent_path)
import fluxonium
import time
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

#%%

""" Parameters """

ec = 1
ej = 17.63 #value Earnest et al 17.63
phi_bias = 0.0
n_fock = 700 # standard 1000
log_el = np.linspace(-4, -2, 10)
el = 10**(log_el)
phi_eg = np.zeros([len(el)], dtype=complex)
s_qp_eg = np.zeros([len(el)], dtype=complex)
a = destroy(n_fock) 

#%%

""" Loop """

start = time.time()

for k in range(0, len(el)):
    flx = fluxonium.Fluxonium(ec, ej, el[k], phi_bias)
    eig_en, eig_vec = flx.eigenstates(n_fock)
    phi = flx.r_phi*(a + a.dag())
    s_qp = (phi/2).sinm()
    phi_eg[k] = np.abs(phi.matrix_element(eig_vec[1], eig_vec[0]))
    s_qp_eg[k] = np.abs(s_qp.matrix_element(eig_vec[1], eig_vec[0]))
    print('Iteration = ' + str(k))

end = time.time()

print(f'Computation time: {end-start} s')

#%%

""" Matrix elements plot """

fig_mat_el, ax1 = plt.subplots(figsize = (6,6))
ax1.semilogx(el, phi_eg, 'o', color = 'red', \
    label = '$\\langle e | \phi | g \\rangle$')
ax1.semilogx(el, s_qp_eg, 'D', color = 'blue', linewidth = 2.0, \
    label = '$\\langle e | \sin(\phi/2) | g \\rangle$')
ax1.set_xlabel('$E_L/E_C$', fontsize = 20)
ax1.legend(fontsize = 16)
plt.title('Matrix elements', fontsize = 20)
plt.show()

#%%
