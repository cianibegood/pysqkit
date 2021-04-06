"""
Analysis of the flux noise sensitivity of the fluxonium qubit
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

log_el = np.linspace(-4, -2, 10)
el = 10**(log_el)
n_fock = 700
t_phi = np.zeros([len(el)], dtype=complex)
eig_en = np.zeros([6, len(el)], dtype=complex)
a_phi_ext = 4*np.pi**2*(3*1e-6)**2
delta_phi = 0.1
phi_bias = 0.0
ec = 1
ej = 17.63 #value Earnest et al 17.63

#%%

""" Qutip operators """

a = destroy(n_fock)
id1 = qeye(n_fock)

#%%

""" Loop """

start = time.time()

for k in range(0, len(el)):
    flx = fluxonium.Fluxonium(ec, ej, el[k], phi_bias)
    eig_0 = flx.eigenenergies(n_fock)[0:6]
    eig_en[:, k] = eig_0
    delta_0 = eig_0[1] - eig_0[0]
    flx = fluxonium.Fluxonium(ec, ej, el[k], phi_bias + delta_phi)
    eig_p = flx.eigenenergies(n_fock)[0:6]
    delta_p = eig_p[1] - eig_p[0]
    flx = fluxonium.Fluxonium(ec, ej, el[k], phi_bias - delta_phi)
    eig_m = flx.eigenenergies(n_fock)[0:6]
    delta_m = eig_m[1] - eig_m[0]
    d2_delta = (delta_p + delta_m - 2*delta_0)/(delta_phi**2)
    t_phi[k] = (a_phi_ext*np.abs(d2_delta))**(-1)
    print('Iteration = ' + str(k))

end = time.time()

print(f'Computation time = {end - start} s')


# %%

""" Plot dephasing """

colors = ['darkorange', 'darkslateblue', 'lightgreen', 'forestgreen', \
    'dimgray', 'darkgray']

ec_value = 2.5 #[GHz]
fig_t_phi = plt.figure(figsize=(6,6))
plt.loglog(el, t_phi/(ec_value*10**9), 'o', markersize = 8, color = 'darkorange', \
    label='$T_{\phi J}$')
plt.xlabel('$E_L/E_C$', fontsize = 20)
plt.ylabel('$T_{\phi}[s]$', fontsize = 20)
plt.title('Dephasing time', fontsize = 20)
plt.show(block = False)


# %%

""" Loglogplot plot dephasing """

fig_eig_scaling1 = plt.figure(figsize = (8,6))
for k in range(0, 6):
    plt.loglog(el, eig_en[k, :] - eig_en[0, :], color = colors[k], \
        linewidth = 2.0)
plt.xlabel('$E_L/E_C$', fontsize = 20)
plt.ylabel('$E_k/E_C$', fontsize = 20)
plt.title('Energy levels', fontsize = 20)
plt.show(block = False)

""" Interestingly they are straight parallel lines in the loglog plot,
but something funny happens for low EL/EC """
# %%
