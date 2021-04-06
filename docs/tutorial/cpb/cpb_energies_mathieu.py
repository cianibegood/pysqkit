""" Energy levels of the CPB and the transmon approximation """

#%%

import numpy as np
from scipy import special
import matplotlib
import matplotlib.pyplot as plt
import sys
from pathlib import Path
parent_path = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_path)
import cpb
import time
matplotlib.rcParams['mathtext.fontset'] = 'cm'
from functools import partial
import itertools
import qutip
import importlib
import os

def kf(m, ng):
    y = 0
    for l in (-1, 1):
        print(l)
        y += np.mod(np.round(2*ng + l/2), 2)*(np.round(ng) + \
            l*(-1)**m*((m + 1)//2))
        print((m + 1)//2)
    print(y)
    return y


def e_cpb(ec, ej, ng, m):
    # Mathieu characteristi value in python seems to work only
    #with integer first argument. The same writing gives always the correct
    #result in Mathematica (see CPBmatheiu.nb)
    ng_eq = ng
    ng_eq = ng - np.floor(ng)
    if ng_eq > 0.5:
        ng_eq = 1 - ng_eq
    y = ec*special.mathieu_a(m + 1 - np.mod(m + 1, 2) + 2*ng*(-1)**m, \
        -ej/(2*ec))
    return y

#%%

""" Data """

ec_ref = 0.25 # [GHz]
ec = 1 
ej = 20
ng = 0.0 #putting ng non integer gives nan in Mathieu function
qubit = cpb.CPB(ec, ej, ng, ec_ref)
n_charges = 100
n_fock = 10
n_order = 6


# %%

""" Exact and approximate eigenenergies """

evals = qubit.eigenenergies(n_charges)
lev = 2
print(evals[lev] - evals[0])
print(e_cpb(ec, ej, ng, lev) - e_cpb(ec, ej, ng, 0))

# %%
for k in (-1, 1):
    print(k)


# %%
