#-----------------------------------------------------------------------------
# Script to plot the result of the scans for a Ficheux-like gate between
# a fluxonium and a transmon as a function of drive frequency and 
# drive strength 
#-----------------------------------------------------------------------------

#%%
import numpy as np 
import matplotlib.pyplot as plt
import json
import scipy.interpolate
from matplotlib.colors import LogNorm
from pylab import cm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

#%%
# Import data 

with open("data_scan_drive_freq_strength.txt", "r") as fp:
    data = json.load(fp)

#%%
save = True
plot_setup = {'fs': 20, 'ls':12}
n_points = len(data)
freq_d = np.zeros(n_points)
eps_d = np.zeros(n_points)
inf_gate = np.zeros(n_points)
l1 = np.zeros(n_points)
for k in range(0, n_points):
    freq_d[k] = data[k][0][0]
    eps_d[k] = data[k][0][1]
    inf_gate[k] = data[k][1]['gate_infidelity']
    l1[k] = data[k][1]['leakage']
freq_d_vec, eps_d_vec = np.linspace(freq_d.min(), \
    freq_d.max(), 100), np.linspace(eps_d.min(), eps_d.max(), 100)
freq_d_vec, eps_d_vec = np.meshgrid(freq_d_vec, eps_d_vec)
# Interpolate
rbf_infidelity = scipy.interpolate.Rbf(freq_d, eps_d, inf_gate, function='linear')
rbf_leakage = scipy.interpolate.Rbf(freq_d, eps_d, l1, function='linear')
inf_gate_rbf = rbf_infidelity(freq_d_vec, eps_d_vec)
l1_rbf = rbf_leakage(freq_d_vec, eps_d_vec)
inf_min = np.min(inf_gate)
inf_max = np.max(inf_gate)
l1_min = np.min(l1)
l1_max = np.max(l1)

#%%

def plot_scans(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    plot_setup: dict, 
    title: str,
    filename: str,
    save: bool
) -> None:
    color_map = 'seismic'
    aspect = (np.max(x) - np.min(x))/(np.max(y) - np.min(y))
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(z, vmin=np.min(z), vmax=np.max(z), 
                   origin='lower', extent=[np.min(x), np.max(x), 
                   np.min(y), np.max(y)], 
                   cmap=color_map, aspect=aspect, 
                   norm=LogNorm(vmin=np.min(z), vmax=np.max(z)))
    x_labels = np.round(np.linspace(np.min(x), np.max(x), 5), 3)
    y_labels = np.round(np.linspace(np.min(y), np.max(y), 5), 5)
    ax.set_xlabel('$\omega_d/2 \pi  \, (\mathrm{GHz})$', 
                  fontsize=plot_setup['fs'])
    ax.set_ylabel('$| \\epsilon_d |$', fontsize=plot_setup['fs'])
    ax.set_xticks(x_labels)
    ax.set_yticks(y_labels)
    ax.tick_params(labelsize=plot_setup['ls'])
    ax.ticklabel_format(axis='y', style='sci')
    cax = fig.add_axes([ax.get_position().x1 + \
        0.05, ax.get_position().y0 , 0.05, ax.get_position().height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=plot_setup['ls'])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(title, fontsize=plot_setup['fs'], rotation=270)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.show()
    if save:
        fig.savefig('tmp/' + filename, bbox_inches='tight')
    

# %%
title_inf = '$1 - f_{\mathrm{gate}}$'
file_inf = 'scan_drive_freq_strength_infidelity.pdf'
plot_scans(freq_d_vec, eps_d_vec, inf_gate_rbf, plot_setup, 
           title_inf, file_inf, save)


# %%
title_leak = '$L_1$'
file_leak = 'scan_drive_freq_strength_leakage.pdf'
plot_scans(freq_d_vec, eps_d_vec, l1_rbf, plot_setup, 
           title_leak, file_leak, save)






# %%
