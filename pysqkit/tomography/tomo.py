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
        store_outputs = False):
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
    
    def simu(state_init):
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
                           pulse_drive, jump_list, "mesolve")
                    
        res = result.states[-1]
        return res


    def _index_to_label(self, index: int):  
        '''returns the label (ie the states of the different qudits
         in the system) from the index'''
        return n_th(self.nb_levels, index)
    
        
    def _label_to_index(self, label: Union[int, Iterable[int]]):
        if isinstance(label, int):
            label = [label]
            
        nb_levels = self.nb_levels
        if len(label) != self._n_qubits:
            raise ValueError("Error! : The length of the label doesn't match ;" \
                + " it should be of length" + str(len(nb_levels)))
        
        for k in range(0, self._n_qubits):
            if label[k] >= self._nb_levels[k]:
                raise ValueError("Label exceeds the dimension of qubit " \
                    + str(k))  
        else:
            return index_from_label(self.nb_levels, label)
    
    def _ket_index(self, n: int):
        return  self._table_states[n] #self._table_states[n].unit()
    def _bra_index(self, n: int):
        return  self._table_states[n].dag()
    def _dm_index(self, n: int):
        return self._ket_index(n) * self._bra_index(n)
    
    #different representations of states from label
    def _ket_label(self, label):
        return self._ket_index(self._label_to_index(label))
    def _bra_label(self, label):
        return self._bra_index(self._label_to_index(label))
    def _dm_label(self, label):
        return self._ket_label(label) * self._bra_label(label)
    
    # |n><m|
    def _rho_nm(self, n: int, m: int):
        return self._ket_index(n)*self._bra_index(m)
    def _rho_j(self, j: int):
        m = j%self.d
        n = j//self.d
        
        return self._rho_nm(n,m)
    
    def _gate_from_simu(self, state_init): 
        '''param should contain 'simu' that runs simu from a dict with parameters and init  ;
        simu should onlytake init as Qobj'''
                   
        return self.simu(state_init)
    
    def to_lambda(
        self, 
        input_states=List[np.ndarray],
        output_states=List[np.ndarray], 
        draw_lambda=False, 
        as_qobj=False
    ):
            
        #clean label lists
        # A.C.: This is not system agnostic and assumes it is 2-qubit
        # A.C.: we could define computational states in the QubitSystem class.
        if input_states=="comp_states":
            in_labels = [(0,0), (0,1), (1,0), (1,1)]
        elif input_states=="all":
            in_labels = [self._index_to_label(k) for k in range(self.d)]
        else:
            raise ValueError("Unsupported input states.")

        if output_states == "comp_states":
            out_labels = [(0,0), (0,1), (1,0), (1,1)]
        elif output_states == "all":
            out_labels = [self._index_to_label(k) for k in range(self.d)]
        else:
            raise ValueError("Unsupported output states.")
            
        #assert isinstance(in_labels, list)
        #assert isinstance(out_labels, list)
        
        for lbl in in_labels:
            assert len(lbl) == len(self.nb_levels)
        for lbl in out_labels:
            assert len(lbl) == len(self.nb_levels)
            
            
        #now transform into lists of indices
        in_ind = []
        out_ind = []
        for lbl in in_labels:
            in_ind.append(self._label_to_index(lbl))
        for lbl in out_labels:
            out_ind.append(self._label_to_index(lbl))
            
        
        #trick to avoid redundant calculations
        key = "lambda"
        key += str(hash(self.system))
        key += str(np.sort(in_ind))
        key += str(np.sort(out_ind))
        
        if key in self.param_syst.keys(): #if already processed
            lambda_mat = self.param_syst[key]
            # print("Used stored for lambda")
            if draw_lambda:
                draw_mat(lambda_mat, "\lambda")  
            if as_qobj:
                return qtp.Qobj(inpt = lambda_mat, dims = [rho_prime_i.dims, self._rho_nm(out_ind[n_j], out_ind[m_j]).dims]) 
            else:
                return lambda_mat

            
        #function to calculate rho_primes
        def rho_prime(n,m):#n,m are indices in the table_states ie in the order of all labels
            if n == m:
                return self.gate([[1, n]])
            else :
                n_n_prime = self.gate([[1, n]])
                m_m_prime = self.gate([[1, m]])
                plus_plus_prime = self.gate([[1/np.sqrt(2), n], [1/np.sqrt(2), m]])
                minus_minus_prime = self.gate([[1/np.sqrt(2), n], [1j/np.sqrt(2), m]])
                
                return plus_plus_prime + 1j*minus_minus_prime - (1+1j)/2 * n_n_prime - (1+1j)/2 * m_m_prime
    
        d = self.d
        #skeleton
        lambda_mat = np.zeros((len(in_ind)**2 , len(out_ind)**2))*1j
    
        #filling
        for i in range(len(in_ind)**2):
            #we range over the pairs of in_labels
            n_i, m_i = n_th([len(in_ind),len(in_ind)], i)
            
            rho_prime_i = rho_prime(in_ind[n_i], in_ind[m_i])
            
                
            for j in range(len(out_ind)**2):
                #we range over the pairs of out_labels
                n_j, m_j = n_th([len(out_ind),len(out_ind)], j)
                
                lambda_mat[i,j] = np.trace(rho_prime_i.full().dot(
                                                self._rho_nm(out_ind[n_j], out_ind[m_j]).dag().full())  
                                        )
                                        
        #once processed, we save it for later (part of trick from above)
        self._param_syst[key] = lambda_mat
        
        
        if draw_lambda:
            draw_mat(lambda_mat, "\lambda")  
        
        if as_qobj:
            return qtp.Qobj(inpt = lambda_mat, dims = [rho_prime_i.dims, self._rho_nm(out_ind[n_j], out_ind[m_j]).dims]) 
        else:
            return lambda_mat
    
    def to_qudit_ptm(
        self, 
        input_states=List[np.ndarray], 
        output_states=List[np.ndarray], 
        as_qobj=False
    ):
        pass
        