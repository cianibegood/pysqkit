""" 
Rabi oscillations in the schroedinger pictures without RWA 
""" 
#%%
import numpy as np 
import qutip as qtp 
from pysqkit.solvers import solvkit

# %%
""" 
Problem setup 
"""

omega = 1.0 
hamil0 = omega/2*qtp.sigmaz()
drive = [qtp.sigmax()]
state_in = qtp.basis(2, 0)
omega_d = 1.0 #drive frequency
omega_rabi = 0.1
tlist = np.linspace(0, 100, 1000)
pulse = [omega_rabi*np.cos(omega_d*tlist)]
jump_op = []
solver = "mesolve"

# %%
"""
Time evolution
"""

output = solvkit.integrate(tlist, state_in, hamil0, drive, 
pulse, jump_op, solver)

# %%
