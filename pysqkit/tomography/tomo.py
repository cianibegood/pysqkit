from typing import Tuple, Optional, Dict, Iterable, Union, Callable, List
import os
import sys

import time
import datetime

import qutip as qtp 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg as la
import cmath

# import pysqkit
from pysqkit.solvers.solvkit import integrate
from pysqkit.systems.system import QubitSystem
from pysqkit.util.linalg import hilbert_schmidt
from pysqkit.util.transformations import iso_basis


from mpl_toolkits.axes_grid1 import make_axes_locatable
   
   
##Tools
#general

def n_th(maxs, n):  
    '''returns n-th tuple with the i_th term varying between 0 and maxs[i]'''
    temp = np.zeros(len(maxs))
    for i in range(0, len(temp)):
        temp[i] = (n//np.prod(maxs[i+1:]))%np.prod(maxs[i])
    
    res = [int(k) for k in temp]
    return res
    
def index_from_label(maxs, label):
    return int(np.sum([label[i] * np.prod(maxs[i+1:])  \
        for i in range(len(label))]))


##  Tomo env class 
    
    
class TomoEnv:   
    def __init__(
        self,
        system: QubitSystem,
        jump_op = [],
        store_outputs = False,
        options: qtp.solver.Options=None
        ):
            '''  Either system is not None and it's all we need OR 
            system is None and all the rest must be defined
            
            table_states is None if we want to take the bare basis'''
            
            self.store_outputs = store_outputs
            self.nb_gate_call = 0
            
            
            self._nb_levels = [qubit.dim_hilbert for qubit in system.qubits]
            self._n_qubits = len(self._nb_levels)
            self._d = int(np.prod(self._nb_levels))
            self._system = system
            self._jump_op = jump_op
                
            self._table_states = [system.state(n_th(self._nb_levels, n), \
                as_qobj = True)[1] for n in range(self._d)] 
            
            self._options = options
                           
    @property
    def nb_levels(self):
        return self._nb_levels
    
    @property
    def n_qubits(self):
        return self._n_qubits
    
    @property
    def d(self):
        return self._d
        
    @property
    def system(self): #what characterizes the env
        return self._system
    
    @property
    def jump_op(self): #what characterizes the env
        return self._jump_op
    
    def simu(self, state_init):
        tlist = [qubit.drives[drive_key].params['time'] for \
            qubit in self._system for drive_key in qubit.drives.keys()][0] 
        hamil0 = self._system.hamiltonian(as_qobj=True)
                    
        hamil_drive = []
        pulse_drive = []
                    
        for qubit in self._system:
            if qubit.is_driven:
                for label, drive in qubit.drives.items():
                    hamil_drive.append(drive.hamiltonian(as_qobj=True))
                    pulse_drive.append(drive.eval_pulse())
                    
        jump_list = self._jump_op 
                    
        result = integrate(tlist*2*np.pi, state_init, hamil0, hamil_drive,
                           pulse_drive, jump_list, "mesolve", options=self._options)
                    
        res = result.states[-1]
        return res   
    
    def evolve_hs_basis(
        self,
        i: int,
        input_states: List[np.ndarray],
        hs_basis: Callable[[int, int], np.ndarray]
    ) -> np.ndarray:
        
        """
        It returns the action of the quantum operation associated
        with the time-evolution on the i-th element of a Hilbert-Schmidt
        basis define via the function hs_basis with computational 
        basis states in input_states.
        """

        d = len(input_states)

        dims_qobj = self._table_states[0].dims

        basis_i = hs_basis(i, d)
        eigvals, eigvecs = np.linalg.eig(basis_i)
        evolved_basis_i = 0
        for n in range(0, d):
            iso_eigvec = 0
            for m in range(0, d):
                iso_eigvec += eigvecs[m, n]*input_states[m]
            iso_eigvec_qobj = qtp.Qobj(inpt=iso_eigvec, dims=dims_qobj)
            rho_iso_eigvec_qobj = iso_eigvec_qobj*iso_eigvec_qobj.dag()
            evolved_iso_eigvec = self.simu(rho_iso_eigvec_qobj)
            evolved_basis_i += eigvals[n]*evolved_iso_eigvec[:, :]
        return evolved_basis_i
    
    def to_super( 
        self, 
        input_states: List[np.ndarray], 
        hs_basis: Callable[[int, int], np.ndarray],
        as_qobj=False
    ):
        """

        """

        d = len(input_states)
        superoperator = np.zeros([d**2, d**2], dtype=complex)
        basis = [] 
        for i in range(0, d**2):
            basis.append(iso_basis(i, input_states, hs_basis))
        
        for i in range(0, d**2):
            evolved_basis = self.evolve_hs_basis(i, input_states, hs_basis)
            for k in range(0, d**2):
                superoperator[k, i] = hilbert_schmidt(basis[k], evolved_basis)
        
        return superoperator

        
