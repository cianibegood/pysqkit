""" Module with functions to study a system of two 
capacitively and inductively coupled flux qubits """

import numpy as np 
from scipy import constants
from scipy.linalg import sqrtm
import scipy.linalg as la
import qutip
from math import factorial, sqrt
import random


def v_pot(el, ej, phi_x, phi_jx, phi):
    """ Potential of a flux qubit """
    y = el*phi**2/2 - ej*np.abs(np.cos(phi_jx/2))*np.cos(phi + phi_x)
    return y

class FluxQubitSystem():

    """ The class allows to study a system of n capacitively 
    and inductively coupled flux qubits (or fluxonia). 
    It implements full numerical diagonalization using qutip when we have
    two flux qubits (to be done for n). """

    def __init__(self, ec_ref, ec, el, ej, phi_x, phi_jx, nfock):

        """ec_ref is a reference charging
        energy in GHz (ec_ref/h). All energies are in 
        units of ec_ref. ec and el are the charging energy
        and inductive energies respectively, given as matrices. 
        ej is a vector with the Josephson energies of each qubit. 
        phi_x, phi_jx are vectors with bias phases for each qubit. nfock
        is the number of Fock states for each qubit to be used
        in qutip. """
        
        self.n = len(ej) #number of flux qubits
        self.ec_ref = ec_ref
        self.ec0 = 1 #ec_ref is taken as unit
        self.ec = ec 
        self.ec_inv = np.linalg.inv(self.ec)
        #ec_inv_sqrt is needed in case we want to diagonalize the kinetic term
        self.ec_inv_sqrt = sqrtm(self.ec_inv) 
        self.el = el
        self.ej = ej
        self.ej_eff = ej*np.abs(np.cos(phi_jx/2))       
        self.phi_x = phi_x 
        self.phi_jx = phi_jx
        self.nfock = nfock
        self.c = 1e6*constants.elementary_charge**2/2*np.linalg.inv(ec)/\
            constants.h/ec_ref #capacitance matrix fF
        self.l = constants.h/(16*np.pi**2*constants.elementary_charge**2*\
            self.ec_ref)*np.linalg.inv(el) #inductance matrix nH
        #s_mat is the transformation matrix of the canonical transformation
        #that diagonalizes the kinetic term
        self.s_mat = sqrtm(ec)
        x = np.linspace(-2, 2, 1000)
        self.u_min_list = np.zeros(self.n)
        for k in range(0, self.n):
            u = v_pot(self.el[k, k], self.ej[k], self.phi_x[k], \
                self.phi_jx[k], x)
            self.u_min_list[k] = np.min(u) #minima of each fq potential
        self.u_min = np.sum(self.u_min_list)

    """ All the Hamiltonians are for two flux qubits only """

    def build_a(self):

        """ This method constructs the annihilation operators (for two
        flux qubits """

        a1 = qutip.tensor(qutip.destroy(self.nfock), \
            qutip.qeye(self.nfock))
        a2 = qutip.tensor(qutip.qeye(self.nfock), \
            qutip.destroy(self.nfock))
        return a1, a2
    
    def h(self):

        """ It returns the Hamiltonian in the Fock basis """

        if self.n != 2:
            raise Exception("The flux qubits must be 2")
        a1, a2 = self.build_a()
        ide = qutip.tensor(qutip.qeye(self.nfock), \
            qutip.qeye(self.nfock))
        phi1 = (2*self.ec[0, 0]/self.el[0, 0])**(1/4)*(a1 + a1.dag())
        phi2 = (2*self.ec[1, 1]/self.el[1, 1])**(1/4)*(a2 + a2.dag())
        n1 = 1j/2*(self.el[0, 0]/(2*self.ec[0, 0]))**(1/4)*(a1.dag() - a1)
        n2 = 1j/2*(self.el[1, 1]/(2*self.ec[1, 1]))**(1/4)*(a2.dag() - a2)
        e_kin = 4*self.ec[0, 0]*n1**2 + 4*self.ec[1, 1]*n2**2 + \
            8*self.ec[0, 1]*n1*n2
        phi_shift1 = phi1 + self.phi_x[0]*ide
        phi_shift2 = phi2 + self.phi_x[1]*ide
        v = self.el[0, 0]/2*phi1**2 - \
            self.ej[0]*np.abs(np.cos(self.phi_jx[0]/2))*phi_shift1.cosm()
        v += self.el[1, 1]/2*phi2**2 - \
            self.ej[1]*np.abs(np.cos(self.phi_jx[1]/2))*phi_shift2.cosm()
        v += self.el[0, 1]*phi1*phi2
        v -= self.u_min*ide #subtract minimum energy of uncoupled potentials
        return e_kin + v    
    
    
    def hsq(self, k):

        """ It returns the single qubit Hamiltonian of qubit k
         in the Fock basis. """

        a = qutip.destroy(self.nfock)
        ide = qutip.qeye(self.nfock)
        phi = (2*self.ec[k, k]/self.el[k, k])**(1/4)*(a + a.dag())
        n = 1j/2*(self.el[k, k]/(2*self.ec[k, k]))**(1/4)*(a.dag() - a)
        e_kin = 4*self.ec[k, k]*n**2
        phi_shift = phi + self.phi_x[k]*ide
        v = self.el[k, k]/2*phi**2 - \
            self.ej_eff[k]*phi_shift.cosm()
        v -= self.u_min_list[k]*ide
        return e_kin + v
    
    def hsq_eigbasis(self, k, nlev):

        """ It returns the first nlev eigenkets of the single qubit 
        Hamiltonian of qubit k in the Fock basis """
        
        h = self.hsq(k)
        ekets = h.eigenstates()[1][0:nlev]
        return ekets 
    
    def hsq_eigvals(self, k, nlev):

        """ It returns the first nlev eigenenergie of the single qubit 
        Hamiltonian of qubit k in the Fock basis as a diagonal 
        quantum object in qutip """

        h = self.hsq(k)
        evals = h.eigenenergies()[0:nlev]
        y = qutip.Qobj(np.diag(evals))
        return y
    
    


