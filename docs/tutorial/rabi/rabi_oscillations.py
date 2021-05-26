""" 
Rabi oscillations in the schroedinger pictures without RWA 
""" 
#%%
import numpy as np 
import matplotlib.pyplot as plt
import qutip as qtp 
from pysqkit.solvers import solvkit
from matplotlib import rcParams 
rcParams['mathtext.fontset'] = 'cm'



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

def pulse_shape(
    t: float, 
    args: dict
    ) -> complex:
    return args["w_rabi"]*np.cos(args["w_drive"]*t)

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
"""
We construct the Hamiltonian as QobjEvo
"""

hamil = qtp.QobjEvo([hamil0, [drive[0], pulse_shape]], \
    args={"w_rabi": omega_rabi, "w_drive": omega_d}, tlist=tlist)
# To access hamil at a given time t
hamil(t=100)

#%%
"""
The Liouvillian can be constructed as
"""
liouv_rabi = qtp.liouvillian(hamil, jump_op)
#liouv_rabi.compile()
#To acces liouv_rabi at a given time t
liouv_rabi(t=100)
chi_map = (tlist[-1]*liouv_rabi(0)).expm()
kraus_map = qtp.to_kraus(chi_map)
rho_in = qtp.ket2dm(state_in)
rho_f = kraus_map[0]*rho_in*kraus_map[0].dag() + \
    kraus_map[1]*rho_in*kraus_map[1].dag()


#%%
"""
Obtaining the CPTP map associated with the evolution. The map 
is given in terms of Kraus operators evaluated at each time step. 
"""

rabi_map = solvkit.cptp_map(tlist, hamil0, drive, [pulse_shape], jump_op, \
    args={"w_rabi": omega_rabi, "w_drive": omega_d})

# %%
"""
Check that final state is consistent 
"""
rho_in = qtp.ket2dm(state_in)
n_times = len(tlist) - 1
# n_times = 0
rho_f = rabi_map[n_times][0]*rho_in*rabi_map[n_times][0].dag() + \
    rabi_map[n_times][1]*rho_in*rabi_map[n_times][1].dag() + \
        rabi_map[n_times][2]*rho_in*rabi_map[n_times][2].dag() + \
            rabi_map[n_times][3]*rho_in*rabi_map[n_times][3].dag()
print(rho_f)


# %%
