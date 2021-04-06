"""
Energy levels and eigenfuntions for the fluxonium qubit
"""

#%%
import numpy as np
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

""" Parameters from Earnest et al "Realization of a Λ system with 
metastable states of a capacitively-shunted fluxonium" """

ec = 1    # unit fixed to 1
ej = 17.63 #value Earnest et al 17.63
el = 0.52 #value Earnest et al 0.52
phi_bias = -0*0.02*2*np.pi # value from Earnest et al -0.02*2*np.pi
n_fock = 100 #1000
n_points = 1000 #flux points for wave function plot
n_lev = 10
delta = 5 #range wave function plot
n_fock_wave = 150 # higher than 160 it crashes when calculating wave function

flx = fluxonium.Fluxonium(ec, ej, el, phi_bias)

#%%

""" Fluxonium potential """

phi_vec = np.linspace(-delta*np.pi, delta*np.pi, n_points)
v_flx = flx.potential(phi_vec)


# %%

""" Plot fluxonium potential """

fig_pot, ax_pot = plt.subplots(figsize = (8, 6))
ax_pot.plot(phi_vec/np.pi, v_flx, color = 'darkviolet', linewidth = 2.0)
ax_pot.set_xlabel('$\phi/\pi$', fontsize = 20)
ax_pot.set_ylabel('$U/E_C$', fontsize = 20)
ax_pot.set_title('Potential', fontsize = 20)
plt.show()

# %%

""" Eigenenergies and eigenvectors (in Fock basis) """

eig_en, eig_vec = flx.eigenstates(n_fock)

# %%

""" Wave functions """

psi = np.zeros([n_lev, n_points], dtype = complex)
for k in range(0, n_lev):
    for m in range(0, n_points):
        psi[k, m] = fluxonium.wave_function(flx.m, flx.omega0, 1,\
            phi_vec[m]/(np.sqrt(2)*flx.r_phi), eig_vec[k][0: n_fock_wave])

# %%

""" Plot wave functions """

lev = 1
shift = eig_en[lev]
ampl = flx.ej
fig_wf, ax_wf = plt.subplots(figsize = (8,6))
# Sqrt(pi)?
ax_wf.plot(phi_vec, ampl*psi[lev, :] + shift, \
    color = 'darkorange', linewidth = 2.0)
ax_wf.plot(phi_vec, v_flx, color = 'darkviolet', linewidth = 2.0)
ax_wf.set_xlabel('$\phi_-/\sqrt{\pi}$', fontsize = 20) #sqrt(pi) label?
ax_wf.set_ylabel('$\Psi$', fontsize = 20)
ax_wf.set_title('Wave function ' + str(lev) , fontsize = 20)
plt.grid(linestyle = '--')
plt.show()


# %%


# %%
