""" It contains functions and classes for the study of the CPB """

import numpy as np
import math
import scipy.linalg as la
from scipy import constants
import qutip 

class CPB:

    """ The class is initialized with the parameters of the CPB and
    a reference charging energy that has to be given in GHz 
    (units of h). ec and ej must be given in units of ec_ref. """

    def __init__(self, ec, ej, ng, ec_ref):
        self.ec = ec
        self.ej = ej
        self.ng = ng
        self.ec_ref = ec_ref # reference charging energy in GHz (units of h)
        self.c = constants.elementary_charge**2/2*1/ec*\
            1/(constants.h*ec_ref)*10**6 # capactiance matrix [fF]
        self.lj = 1/(4*np.pi**2)*1/ej*1/(constants.h*ec_ref)*\
            (constants.h/(2*constants.elementary_charge))**2 #ind matrix [nH]
        self.omega_p = np.sqrt(8*ec*ej)
        self.phi_zpf = (2*ec/ej)**(1/4)
    
    def h_cpb(self, n_charges):

        """ Hamiltonian of the CPB in charge basis """

        ide = np.identity(2*n_charges + 1)
        n0 = ncp(n_charges)
        jj_t = josephson_tunnel(n_charges)
        cos_jt = 1/2*(jj_t + jj_t.conj().T)
        n_tot = n0 + self.ng*ide
        kin = 4*self.ec*n_tot.dot(n_tot)
        u = self.ej*cos_jt
        h = kin - u
        return h
    
    def eigenenergies(self, n_charges):

        """ Eigenenergies of the CPB in charge basis """

        h = self.h_cpb(n_charges)
        y1 = np.linalg.eigvals(h)
        idx = y1.argsort()[::1]   
        y1 = y1[idx]
        return y1
    
    def h_cpb_approx(self, n_order, n_fock):

        if np.mod(n_order, 2) != 0:
            raise Exception("The order must be even.")

        ide = qutip.qeye(n_fock)
        a = qutip.destroy(n_fock)
        h = self.omega_p*(a.dag()*a + ide/2) - self.ej*ide
        phi = self.phi_zpf*(a + a.dag())
        n_eff = int(n_order/2)
        for n in range(2, n_eff + 1):
            h += -self.ej*(-1)**n/(math.factorial(2*n))*phi**(2*n)
        return h
    
    def h_cpb_rwa(self, n_order, n_fock):

        if np.mod(n_order, 2) != 0:
            raise Exception("The order must be even.")

        ide = qutip.qeye(n_fock)
        a = qutip.destroy(n_fock)
        h = self.omega_p*(a.dag()*a + ide/2) - self.ej*ide
        n_eff = int(n_order/2)
        for n in range(2, n_eff + 1):
            h += -self.ej*(-1)**n/(math.factorial(2*n))*\
                pow_op_rwa(2*n, self.phi_zpf*a)
        return h
    
    def h_cpb_next_rwa(self, n_order, n_fock):

        if np.mod(n_order, 2) != 0:
            raise Exception("The order must be even.")

        ide = qutip.qeye(n_fock)
        a = qutip.destroy(n_fock)
        h = self.omega_p*(a.dag()*a + ide/2) - self.ej*ide
        n_eff = int(n_order/2)
        for n in range(2, n_eff + 1):
            h += -self.ej*(-1)**n/(math.factorial(2*n))*\
                pow_op_next_rwa(2*n, self.phi_zpf*a)
        return h


def ncp(n_charges):

    """ Charge operator in the charge basis """

    nop = np.zeros([2*n_charges + 1, 2*n_charges + 1], \
        dtype=complex)
    for k in range(0, 2*n_charges + 1):
        nop[k, k] = -n_charges + k
    return nop

def josephson_tunnel(n_charges):

    """ Josephson tunneling operator in charge basis """

    tun_op = np.zeros([2*n_charges + 1, 2*n_charges + 1], \
        dtype=complex)
    for k in range(0, 2*n_charges):
        tun_op[k + 1, k] = 1
    return tun_op

def pow_op_rwa(n, op):

    """ It returns the operator (op + op^{dag})^n performing a rwa, i.e.,
    neglecting terms with unequal number of annihilation and 
    creation operators. It uses qutip."""

    if np.mod(n, 2) != 0:
            raise Exception("The order must be even. If odd the result is 0.")
    
    y_sum = 0
    for k in range(0, 2**n):
        """ We represent the number in base 2 and convert it first
        as list and then numpy array """
        comb_rep = list(np.base_repr(k))
        comb_rep = list(map(int, comb_rep))
        comb_rep = np.asarray(comb_rep)
        comb = np.zeros(n)
        comb[-len(comb_rep):] = comb_rep
        y = 0
        if np.sum(comb) == int(n/2):
            if comb[n-1] == 1:
                y = op.dag()
            elif comb[n-1] == 0:
                y = op
            for m in range(1, n):
                if comb[n-1-m] == 1:
                    y = op.dag()*y
                elif comb[n-1-m] == 0:
                    y = op*y
            y_sum += y 
    return y_sum

def pow_op_next_rwa(n, op):

    """ It returns the operator (op + op^{dag})^n performing a "next" rwa,
    i.e., neglecting terms for which the number of annihilation and 
    creation operators differ by at most 1. It uses qutip. """

    if np.mod(n, 2) != 0:
            raise Exception("The order must be even. If odd the result is 0.")
    
    y_sum = 0
    for k in range(0, 2**n):
        """ We represent the number in base 2 and convert it first
        as list and then numpy array """
        comb_rep = list(np.base_repr(k))
        comb_rep = list(map(int, comb_rep))
        comb_rep = np.asarray(comb_rep)
        comb = np.zeros(n)
        comb[-len(comb_rep):] = comb_rep
        y = 0
        if np.sum(comb) == int(n/2) or np.sum(comb) == int(n/2)+1 \
            or np.sum(comb) == int(n/2)-1 :
            if comb[n-1] == 1:
                y = op.dag()
            elif comb[n-1] == 0:
                y = op
            for m in range(1, n):
                if comb[n-1-m] == 1:
                    y = op.dag()*y
                elif comb[n-1-m] == 0:
                    y = op*y
            y_sum += y 
    return y_sum







    