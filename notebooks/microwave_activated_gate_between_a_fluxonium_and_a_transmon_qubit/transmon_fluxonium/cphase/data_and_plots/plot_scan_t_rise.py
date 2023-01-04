#-----------------------------------------------------------------------------
# Script to plot the result of the scans for a Ficheux-like gate between
# a fluxonium and a transmon as a function of the rising time
#-----------------------------------------------------------------------------

#%%
import numpy as np 
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

#%%
# Import data 

with open("data_scan_t_rise.txt", "r") as fp:
    data = json.load(fp)

# %%
save = True
n_points = len(data)
t_rise = np.zeros(n_points)
inf_gate = np.zeros(n_points)
l1 = np.zeros(n_points)
for k in range(0, n_points):
    t_rise[k] = data[k][0]
    inf_gate[k] = data[k][1]['gate_infidelity']
    l1[k] = data[k][1]['leakage']
inf_gate = inf_gate[np.argsort(t_rise)]
l1 = l1[np.argsort(t_rise)]
t_rise = np.sort(t_rise)

# %%
plot_setup = {'fs': 20, 'ls':12, 'lw': 2.0}

def plot_scan(
    x: np.ndarray,
    y: np.ndarray,
    plot_setup: dict, 
    y_label,
    filename: str,
    save: bool
) -> None:
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(x, y, linewidth=plot_setup['lw'], color='blue')
    ax.set_xlabel('$t_{\mathrm{rise}}  \, (\mathrm{ns})$', 
                  fontsize=plot_setup['fs'])
    ax.tick_params(labelsize=plot_setup['ls'])
    ax.set_ylabel(y_label, fontsize=plot_setup['fs'])
    if save:
        fig.savefig('tmp/' + filename, bbox_inches='tight')
    plt.show()

#%%
y_label_inf = '$1 - f_{\mathrm{gate}}$'
file_inf = 'scan_t_rise_infidelity.pdf'

plot_scan(t_rise, inf_gate, plot_setup, y_label_inf, file_inf, save)

# %%
y_label_leak = '$L_1$'
file_leak = 'scan_t_rise_leak.pdf'

plot_scan(t_rise, l1, plot_setup, y_label_leak, file_leak, save)


# %%
