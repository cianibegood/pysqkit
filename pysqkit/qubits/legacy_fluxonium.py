"""
Module for the study of the fluxonium qubit
"""

# %%

import numpy as np
from qutip import *
import scipy.linalg
import scipy.special
from math import factorial, sqrt


class Fluxonium():

    """ Class with parameters and methods to study the fluxonium qubit.
    It uses qutip. """

    def __init__(self, ec, ej, el, phi_ext):
        self.ec = ec
        self.ej = ej
        self.el = el
        self.phi_ext = phi_ext  # external phase
        self.omega0 = np.sqrt(8*el*ec)
        self.m = 1/(8*ec)  # equivalent "mass"
        self.r_phi = (2*ec/el)**(1/4)

    def hamiltonian(self, n_fock):
        a = destroy(n_fock)
        id_f = qeye(n_fock)
        h = self.omega0*(a.dag()*a + id_f/2)
        phi_tot = self.r_phi*(a + a.dag()) + self.phi_ext
        h += -self.ej*phi_tot.cosm()
        return h

    def eigenstates(self, n_fock):
        h = self.hamiltonian(n_fock)
        eig_en, eig_vec = h.eigenstates()
        return eig_en, eig_vec

    def eigenenergies(self, n_fock):
        h = self.hamiltonian(n_fock)
        eig_en = h.eigenenergies()
        return eig_en

    def potential(self, phi):
        v = self.el/2*phi**2 - self.ej*np.cos(phi + self.phi_ext)
        return v


def wave_function(mass, omega, hbar, x, psi):
    """ Gives the wave function in position representation at x given
    its Fock states representation """

    coef = np.zeros(psi.shape[0], dtype=complex)
    for n in range(0, len(coef)):
        coef[n] = 1/(sqrt(2**n*factorial(n))) *\
            (mass*omega/(np.pi*hbar))**(1/4)*np.exp(-x**2/2)*psi[n]
    psi_x = np.polynomial.hermite.hermval(x, coef)
    return psi_x
