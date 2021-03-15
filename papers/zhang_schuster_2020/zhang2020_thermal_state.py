"""
We want to analyze the thermal state of the system in
Zhang-Schuster et al "Universal fast flux control of a 
coherent, low-frequency qubit"
"""

#%%

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.special
from math import factorial, sqrt
import sys
from pathlib import Path
parent_path = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_path)
import fluxonium
import pathlib
import time
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

#%%

""" Parameters from from Schuster et al """

ec_ref = 0.479 #GHz
ec = 1    # unit fixed to 1
ej = 3.395/ec_ref 
el = 0.132/ec_ref 
phi_bias = 1/2*2*np.pi 
n_fock = 1000
n_points = 1000
n_lev = 10
delta = 4
n_fock_wave = 150 # higher than 160 it crashes when calculating wave function
save = False

flx = fluxonium.Fluxonium(ec, ej, el, phi_bias)

#Unfinished
