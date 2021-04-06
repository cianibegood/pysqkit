""" 
Rabi oscillations in the schroedinger pictures without RWA 
""" 
#%%
import numpy as np 
import matplotlib.pyplot as plt
import qutip as qtp 

import sys

parent_path = "C:\\Users\\nicol\\Git\\QuTech 2021\\pysqkit" #"..\\..\\.." 
sys.path.append(parent_path)


from pysqkit.solvers import solvkit
# from matplotlib import rcPxarams 
# rcParams['mathtext.fontset'] = 'cm'


def plot_rabi(tlist, sz_avg):
    fs = 24
    ts = 14
    lw = 2.0
    col = 'blue'
    fig, ax = plt.subplots(figsize=[8, 6])
    ax.plot(tlist, sz_avg, color=col, linewidth=lw)
    ax.set_xlabel('$t$', fontsize=fs)
    ax.set_ylabel('$\\langle Z \\rangle$', fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=ts)
    plt.grid(linestyle = '--')
    plt.show()

# %%
""" 
Problem setup 
"""

tlist = np.linspace(0, 3000, 10000)
state_in = qtp.basis(2, 0)
omega = 1.0 
hamil0 = omega/2*qtp.sigmaz()
drive = [qtp.sigmax()] #more drives can be added
omega_d = 1.0 #drive frequency
omega_rabi = 0.01 #by increasing this you clearly see non-RWA effects
tlist = np.linspace(0, 3000, 10000)
pulse = [omega_rabi*np.cos(omega_d*tlist)]
gamma = 0.001 #relaxation rate
jump_op = [np.sqrt(gamma)*qtp.sigmam()]
solver = "mesolve"

#%%
"""
Time evolution
"""

output = solvkit.integrate(tlist, state_in, hamil0, drive, 
    pulse, jump_op, solver)

# %%
""" 
Getting average of pauli z
"""

sz_avg = np.zeros(len(tlist), dtype=float) 
for k in range(0, len(tlist)):
    sz_avg[k] = qtp.expect(qtp.sigmaz(), output.states[k])

# %%
"""
Plot
"""

plot_rabi(tlist, sz_avg)

# %%
