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


from mpl_toolkits.axes_grid1 import make_axes_locatable
   
   
##Tools
#general
def _n_th(maxs, n):  
    '''returns n-th tuple with the i_th term varying between 0 and maxs[i]'''
    temp = np.zeros(len(maxs))
    for i in range(0, len(temp)):
        temp[i] = (n//np.prod(maxs[i+1:]))%np.prod(maxs[i])
    
    res = [int(k) for k in temp]
    return res
    
def _index_from_label(maxs, label):
    return int(np.sum([label[i] * np.prod(maxs[i+1:])  for i in range(len(label))]))
    
#visualisation
def draw_mat(mat, mat_name, vmin = np.NaN, vmax = np.NaN, show = False):
    ''' To represent a matrix, we show its module, real part and imaginary part'''
    
    if np.isnan(vmin):
        vmin = min(np.min(mat.real), np.min(mat.imag))
    if np.isnan(vmax):
        vmax = np.max([np.max(mat.real), np.max(mat.imag)])
        
    fig, ax = plt.subplots(1, 3, figsize = (13,5))

    im0 = ax[0].imshow(np.abs(mat), vmin=vmin, vmax=vmax)
    ax[0].set_title("$| "+ mat_name + "_{ij} |$")
    ax[0].set_xlabel('$i$')
    ax[0].set_ylabel('$j$')


    im1 = ax[1].imshow(mat.real, vmin=vmin, vmax=vmax)
    ax[1].set_title("$Re(  "+ mat_name + "_{ij} )$")
    ax[1].set_xlabel('$i$')
    ax[1].set_ylabel('$j$')


    im2 = ax[2].imshow(mat.imag, vmin=vmin, vmax=vmax)
    ax[2].set_title("$Im(  "+ mat_name + "_{ij} )$")
    ax[2].set_xlabel('$i$')
    ax[2].set_ylabel('$j$')
    
    fig.colorbar(im2, ax=ax.ravel().tolist(), orientation = 'horizontal')
    
    if show:
        plt.show()
    #show and save ??

def draw_mat_mult(mat_list, mat_name_list, vmin = np.NaN, vmax = np.NaN, show = False):
    fig, ax = plt.subplots(len(mat_list), 3, figsize = (12,4*len(mat_list)))
    
    for i in range(len(mat_list)):
        
        if np.isnan(vmin):
            vmin = min(np.min(mat_list[i].real), np.min(mat_list[i].imag))
        if np.isnan(vmax):
            vmax = np.max([np.max(mat_list[i].real), np.max(mat_list[i].imag)])
            
        
        im0 = ax[i, 0].imshow(np.abs(mat_list[i]), vmin=vmin, vmax=vmax)
        ax[i, 0].set_title("$| "+ mat_name_list[i] + "_{ij} |$")
        ax[i, 0].set_xlabel('$i$')
        ax[i, 0].set_ylabel('$j$')


        im1 = ax[i, 1].imshow(mat_list[i].real, vmin=vmin, vmax=vmax)
        ax[i, 1].set_title("$Re(  "+ mat_name_list[i] + "_{ij} )$")
        ax[i, 1].set_xlabel('$i$')
        ax[i, 1].set_ylabel('$j$')


        ax[i, 2].imshow(mat_list[i].imag, vmin=vmin, vmax=vmax)
        ax[i, 2].set_title("$Im(  "+ mat_name_list[i] + "_{ij} )$")
        ax[i, 2].set_xlabel('$i$')
        ax[i, 2].set_ylabel('$j$')

        fig.colorbar(im0, ax=ax[i, :].ravel().tolist(), orientation = 'vertical')
    if show:
        plt.show()
    
    
            
def process_fidelity(env_real, U_ideal, correc, labels_chi_1 = "comp_states"):
    '''both env should have the same size, and the same eigenstates
    
    correc returns the correction that needs to be operated onto U_ideal. It takes env_real as argument
    
        The U_ideal should match the size of the env_real and the correc output should be compatible'''
    
    if labels_chi_1 == "comp_states":
        labels_chi_1 = [(0,0), (0,1), (1,0), (1,1)]
        
    ind_chi_1 = [env_real._label_to_index(label) for label in labels_chi_1]
    
    lambda_real = env_real.fct_to_lambda(in_labels = labels_chi_1, out_labels = labels_chi_1, draw_lambda = False, as_qobj = False)
    
    U_correction = correc(env_real, in_out_labels = labels_chi_1)
    U_ideal_correc = U_correction.conj().T.dot(U_ideal)
     
    env_ideal = TomoEnv(definition_type = 'U',
                        nb_levels = env_real.nb_levels,
                        param_syst = {'U' : U_ideal_correc},
                        table_states = env_real._table_states)
    lambda_ideal = env_ideal.fct_to_lambda(in_labels = labels_chi_1, out_labels = labels_chi_1, draw_lambda = False, as_qobj = False)
    
    assert lambda_real.shape == lambda_ideal.shape
    lambda_tilde = lambda_ideal.T.conj().dot(lambda_real)

    return np.trace(lambda_tilde)/(len(labels_chi_1)**2)
    
def avg_gate_fid(env_real, U_ideal, correc, labels_chi_1 = "comp_states", use_correc_in_L1 = True, mute = False):
    
    d1 = len(labels_chi_1)
    if use_correc_in_L1:
        L1 = env_real.L1(U_ideal = U_ideal, correc = correc, labels_chi_1 = labels_chi_1)
    else:
        L1 = env_real.L1(labels_chi_1 = labels_chi_1)
        
    F_pro = process_fidelity(env_real = env_real, U_ideal = U_ideal, correc = correc, labels_chi_1 = labels_chi_1)
    
    if not mute :
        print("We made ", env_real.nb_gate_call, "gate calls")
    return (d1*F_pro + 1 - L1)/(d1 + 1)
    
def L1_from_scratch(system, U_ideal = None, correc = None, labels_chi_1 = "comp_states", store_outputs = False):
    env = TomoEnv(system = system, store_outputs = store_outputs)
    return env.L1(U_ideal = U_ideal, correc = correc, labels_chi_1 = labels_chi_1)
    
def L2_from_scratch(system, U_ideal = None, correc = None, labels_chi_1 = "comp_states", store_outputs = False):
    env = TomoEnv(system = system, store_outputs = store_outputs)
    return env.L2(U_ideal = U_ideal, correc = correc, labels_chi_1 = labels_chi_1)
    
def process_fidelity_from_scratch(system, U_ideal, correc, labels_chi_1 = "comp_states", store_outputs = False):
    env_real = TomoEnv(system = system, store_outputs = store_outputs)
    
    return process_fidelity(env_real = env_real, U_ideal = U_ideal, correc = correc, labels_chi_1 = labels_chi_1)
    
    
def avg_gate_fid_from_scratch(system, U_ideal, correc, labels_chi_1 = "comp_states", store_outputs = False, use_correc_in_L1 = True, mute = False):
    env_real = TomoEnv(system = system, store_outputs = store_outputs)

    return avg_gate_fid(env_real = env_real, U_ideal = U_ideal, correc = correc, labels_chi_1 = labels_chi_1, use_correc_in_L1 = use_correc_in_L1, mute = mute)
    
    
## Fast versions
def process_fidelity_fast(env_real, U_ideal, correc, labels_chi_1 = "comp_states", assume_U_id_corrected_is_diag = True, zero_threshold = 1e-8):
    '''both env should have the same size, and the same eigenstates
    
    correc returns the correction that needs to be operated onto U_ideal. It takes env_real as argument
    
        The U_ideal should match the size of the env_real and the correc output should be compatible
        
        U_ideal should be diagonal'''
    
    
    if labels_chi_1 == "comp_states":
        labels_chi_1 = [(0,0), (0,1), (1,0), (1,1)]
    ind_chi_1 = [env_real._label_to_index(label) for label in labels_chi_1]
    
    U_correction = correc(env_real, in_out_labels = labels_chi_1)
    U_ideal_correc = U_correction.conj().T.dot(U_ideal)
     
    env_ideal = TomoEnv(definition_type = 'U',
                        nb_levels = env_real.nb_levels,
                        param_syst = {'U' : U_ideal_correc},
                        table_states = env_real._table_states)
    lambda_ideal = env_ideal.fct_to_lambda(in_labels = labels_chi_1, out_labels = labels_chi_1, draw_lambda = False, as_qobj = False)
    
    lambda_ideal_conj = lambda_ideal.T.conj()
    
    res = 0.j
    if assume_U_id_corrected_is_diag:
        for first_ind in range(len(ind_chi_1)):
            for second_ind in range(len(ind_chi_1)):
                n = ind_chi_1[first_ind]
                m = ind_chi_1[second_ind]
        
        #the indices are different for the two matrices because the ideal one is only expressed in chi_1 while the other one takes the general indices
        #Trace of product is : sum over i and k of M_{ik} M'_{ki} = M_{ii} M'_{ii} if one is diagonal
                if np.abs(lambda_ideal_conj[first_ind*len(ind_chi_1) + second_ind, 
                                            first_ind*len(ind_chi_1) + second_ind]) > zero_threshold:
                    res += lambda_ideal_conj[first_ind*len(ind_chi_1) + second_ind, 
                                                first_ind*len(ind_chi_1) + second_ind] * \
                            env_real.lambda_as_function(n*env_real.d + m, n*env_real.d + m)

    else :
        for first_ind_trace in range(len(ind_chi_1)):
            for second_ind_trace in range(len(ind_chi_1)):
                for first_ind_prod in range(len(ind_chi_1)):
                    for second_ind_prod in range(len(ind_chi_1)):
                        n_trace = ind_chi_1[first_ind_trace]
                        m_trace = ind_chi_1[second_ind_trace]
                        
                        n_prod = ind_chi_1[first_ind_prod]
                        m_prod = ind_chi_1[second_ind_prod]
                
                #the indices are different for the two matrices because the ideal one is only expressed in chi_1 while the other one takes the general indices
                #Trace of product is : sum over i and k of M_{ik} M'_{ki}
                if np.abs(lambda_ideal_conj[first_ind_trace*len(ind_chi_1) + second_ind_trace, 
                                                    first_ind_prod*len(ind_chi_1) + second_ind_prod]) > zero_threshold:
                            res += lambda_ideal_conj[first_ind_trace*len(ind_chi_1) + second_ind_trace, 
                                                    first_ind_prod*len(ind_chi_1) + second_ind_prod] * \
                                    env_real.lambda_as_function(n_prod*env_real.d + m_prod, n_trace*env_real.d + m_trace)

    return res/(len(labels_chi_1)**2)
    
def avg_gate_fid_fast(env_real, U_ideal, correc, labels_chi_1 = "comp_states", use_correc_in_L1 = True, mute = False, assume_U_id_corrected_is_diag = True, zero_threshold = 1e-8):
    
    d1 = len(labels_chi_1)
    if use_correc_in_L1:
        L1 = env_real.L1(U_ideal = U_ideal, correc = correc, labels_chi_1 = labels_chi_1)
    else:
        L1 = env_real.L1(labels_chi_1 = labels_chi_1)
        
    F_pro = process_fidelity_fast(env_real = env_real, U_ideal = U_ideal, correc = correc, labels_chi_1 = labels_chi_1, assume_U_id_corrected_is_diag = assume_U_id_corrected_is_diag, zero_threshold = zero_threshold)
    if not mute :
        print("We made ", env_real.nb_gate_call, "gate calls")
    return (d1*F_pro + 1 - L1)/(d1 + 1)
    
def process_fidelity_from_scratch_fast(system, U_ideal, correc, labels_chi_1 = "comp_states", store_outputs = True, assume_U_id_corrected_is_diag = True, zero_threshold = 1e-8):
    env_real = TomoEnv(system = system, store_outputs = store_outputs)
    
    return process_fidelity_fast(env_real = env_real, U_ideal = U_ideal, correc = correc, 
                                    labels_chi_1 = labels_chi_1, 
                                    assume_U_id_corrected_is_diag = assume_U_id_corrected_is_diag,
                                    zero_threshold = zero_threshold)
    
    
def avg_gate_fid_from_scratch_fast(system, U_ideal, correc, labels_chi_1 = "comp_states", store_outputs = True, mute = False, assume_U_id_corrected_is_diag = True, use_correc_in_L1 = True, zero_threshold = 1e-8):
    env_real = TomoEnv(system = system, store_outputs = store_outputs)
    
    return avg_gate_fid_fast(env_real = env_real, U_ideal = U_ideal, correc = correc, 
                            labels_chi_1 = labels_chi_1, 
                            assume_U_id_corrected_is_diag = assume_U_id_corrected_is_diag,
                            mute = mute,
                            zero_threshold = zero_threshold,
                            use_correc_in_L1 = use_correc_in_L1)


        
##  Tomo env class 
    
    
class TomoEnv:   
    def __init__(
        self,
        system = None,
        definition_type: str = None,
        nb_levels: Union[int, Iterable[int]] = None ,
        param_syst = None,
        table_states = None,
        jump_op = [],
        store_outputs = False):
            '''  Either system is not None and it's all we need OR system is None and all the rest must be defined
            
            table_states is None if we want to take the bare basis'''
            
            self.store_outputs = store_outputs
            self.nb_gate_call = 0
            
            if system is None: #old method
            
                assert not definition_type is None
                assert not nb_levels is None
                assert not param_syst is None
                
                
                #def type
                assert definition_type in ['U', 'kraus'] #'2-qubit simu']
                self._definition_type = definition_type
                
                #nb_levels and d
                if isinstance(nb_levels, int):
                    nb_levels = [nb_levels]
                self._nb_levels = nb_levels
                self._d = int(np.prod(nb_levels))
                
                #parameters 
                if self._definition_type == 'U':
                    assert 'U' in param_syst.keys()
                    assert hasattr(param_syst['U'], 'shape') \
                            and (param_syst['U'].shape[0] == self._d) \
                            and (param_syst['U'].shape[1] == self._d)
                    self._carac = param_syst['U']
                            
                elif self._definition_type == 'kraus':
                    assert 'op_list' in param_syst.keys()
                    assert len(param_syst['op_list']) <= self._d**2
                    for op in param_syst['op_list']:
                        assert hasattr(op, 'shape') \
                                and (op.shape[0] == self._d) \
                                and (op.shape[1] == self._d)
                    self._carac = param_syst['op_list']
                
                # elif self._definition_type == '2-qubit simu':
                #     assert 'qb1' in param_syst.keys()
                #     assert 'qb2' in param_syst.keys()
                #     assert 'jc' in param_syst.keys()
                #     
                #     # assert 'get_state_basis' in param_syst.keys()  #function whose only argument is param_syst  
                #     # table_states = param_syst['get_state_basis'](param_syst)
                #     
                #     # assert 'get_h_drive' in param_syst.keys()  #function whose only argument is param_syst
                #     # assert 'get_pulse_drive' in param_syst.keys()  #function whose only argument is param_syst
                #     # assert 'get_jump' in param_syst.keys()  #function whose only argument is param_syst
                #     # assert 'get_tlist' in param_syst.keys()  #function whose only argument is param_syst 
                #     
                #     assert 'simu' in param_syst.keys()  #could be otehrwise but allows to control the output 
                #     #the simu one takes an initial state (of type qobj) and param_syst
                
                    
                    
                self._param_syst = param_syst 
                
                #table_states
                if table_states is None:
                    self._table_states = []
                    for k in range(self._d):
                        self._table_states.append(qtp.fock(self._nb_levels, _n_th(self._nb_levels, k) ))
                else :
                    self._table_states = table_states #should be a list of states in ket form, ordered by ascending label





            else : #def from system, only system and table states if not none will be taken into consideration
            #for now only deals with 2 qubit systems
                self._nb_levels = [qubit.dim_hilbert for qubit in system.qubits]
                self._d = int(np.prod(self._nb_levels))
                self._definition_type = '2system'
                self._carac = system
                
                #table_states
                if table_states is None:
                    self._table_states = [system.state(_n_th(self._nb_levels, n), as_qobj = True)[1] for n in range(self._d)] 
                else:
                    self._table_states = table_states
                
                self._param_syst =  {}
                self._param_syst['system'] = system
                
                def simu(state_init):
                    tlist = [qubit.drives[drive_key].params['time'] for qubit in system for drive_key in qubit.drives.keys()][0]#we assume that it is lways the same
                    hamil0 = system.hamiltonian(as_qobj=True)
                    
                    hamil_drive = []
                    pulse_drive = []
                    
                    for qubit in system:
                        if qubit.is_driven:
                            for label, drive in qubit.drives.items():
                                hamil_drive.append(drive.hamiltonian(as_qobj=True))
                                pulse_drive.append(drive.eval_pulse())
                    
                    jump_list = jump_op #[qubit.dielectric_loss() for qubit in system.qubits]
                    
                    result = integrate(tlist*2*np.pi, state_init, hamil0, hamil_drive, pulse_drive, jump_list, "mesolve")
                    
                    res = result.states[-1]
                    return res #/np.trace(res.full())
                    
                self._param_syst['simu'] = simu
                    
        
#only getters, no setters        
    @property
    def nb_levels(self):   #nb_levels is an int or a list of the number of levels of each system
        return self._nb_levels
    
    @property
    def d(self):
        return self._d
    
    @property
    def param_syst(self):
        return self._param_syst
        
    @property
    def carac(self): #what characterizes the env
        return self._carac
        
    @property
    def definition_type(self):
        return self._definition_type
        
    # @property
    # def table_states(self):
    #     return self._table_states
        
    
#tools 

    def _index_to_label(self, index: int):  
        '''returns the label (ie the states of the different qudits in the system) from the index'''
        return _n_th(self.nb_levels, index)
    
        
    def _label_to_index(self, label: Union[int, Iterable[int]]):
        if isinstance(label, int):
            label = [label]
            
        nb_levels = self.nb_levels
        if len(label) != len(nb_levels):
            print("Error! : The length of the label doens't match ; it should be of length"+str(len(nb_levels)))
            
        else:
            return _index_from_label(self.nb_levels, label)
            


## Preparing the ground
#defining the states
#we choose them normalized
    #different representations of states from index
    def _ket_index(self, n: int):
        return  self._table_states[n].unit()
    def _bra_index(self, n: int):
        return  self._table_states[n].dag().unit()
    def _dm_index(self, n: int):
        return (self._ket_index(n) * self._bra_index(n)).unit()
    
    #different representations of states from label
    def _ket_label(self, label):
        return (self._ket_index(self._label_to_index(label))).unit()
    def _bra_label(self, label):
        return (self._bra_index(self._label_to_index(label))).unit()
    def _dm_label(self, label):
        return (self._ket_label(label) * self._bra_label(label)).unit()
    
    # |n><m|
    def _rho_nm(self, n: int, m: int):
        return (self._ket_index(n)*self._bra_index(m)).unit()
    def _rho_j(self ,j: int):
        m = j%self.d
        n = j//self.d
        
        return self._rho_nm(n,m).unit()
        
    
    # def _basis_rho_nm(nb_levels: Union[int, Iterable[int]] ):
    #     '''nb_levels is an int or a list of the number of levels of each system'''
    #     if isinstance(nb_levels, int):
    #         nb_levels = [nb_levels]
    #     d = np.prod(nb_levels)
    #     
    #     res = []
    #     for k in range(d**2):
    #         res.append(_rho_nm_flat(k , nb_levels))
    # 
    #     return res 
    
#defining the operator basis: 
#we choose it normalized
    @staticmethod
    def _Pauli_gen(j, local_d):#operator P_j for 1 system
        
        if j >= local_d**2:
            print("j selected too big for the local number of levels, j should be less than local_d**2")
            return None
        
        a1 = j//local_d
        a2 = j%local_d
    
        if a1<a2:
            res = np.zeros((local_d, local_d))*0j
            res[a1,a2] = 1
            res[a2,a1] = 1
            return qtp.Qobj(res)
    
        elif a1>a2:
            res = np.zeros((local_d, local_d))*0j
            res[a1,a2] = 1j
            res[a2,a1] = -1j
            return qtp.Qobj(res)
    
        else:#a1=a2
            if a1==0:
                return qtp.qeye(local_d)
    
            elif a1<(local_d-1):
                res = np.zeros((local_d, local_d))*0j
                h_k_lvl_inf = TomoEnv._Pauli_gen(a1*(local_d-1) + a2, local_d-1)  #we take one at same coordinates in lower level basis
                res[:-1, :-1] = h_k_lvl_inf.full()
                return qtp.Qobj(res)
    
            else : #a1 = a2 = lvls-1
                res = np.zeros((local_d, local_d))*0j
                res[:-1, :-1] = qtp.qeye(local_d-1).full()
                res[-1, -1] = 1-local_d
                res = np.sqrt(2/(local_d*(local_d-1)))*res
                return qtp.Qobj(res)
            
        
    def _E_tilde_pauli(self, i):
        
        nb_levels = self.nb_levels
            
        pauli_list = []
        pauli_maxs = [int(max_lvl**2) for max_lvl in nb_levels]
        
        tpl = _n_th(pauli_maxs, i) #tuple of indices of the P_i that will appear in the product defining E_tilde 
                                #(as in sigma_i x sigma_j x sigma_k)
                                #For L levels, there are L^2 _Pauli_gen matrices
                                
        for j in range(len(tpl)):
            ind = tpl[j]
            pauli_list.append(TomoEnv._Pauli_gen(ind, nb_levels[j]).unit())
            
        return qtp.tensor(pauli_list).unit()
     
                            
    def _basis_E_tilde_pauli(self):
        res = []
        for i in range(self.d**2):
            res.append(self._E_tilde_pauli(i).unit())
    
        return res
    
# Defining gates
    def _gate_from_Kraus(self, state_init):
        res = qtp.Qobj(np.zeros(state_init.shape), dims = state_init.dims)
        for op in self.param_syst['op_list']:
            op = qtp.Qobj(op, dims = state_init.dims)
            res+= op*state_init*op.dag()
        
        return res#/np.trace(res.full())
    
    
    def _gate_from_U(self, state_init): 
        '''param should contain U which is a 2D numpy array'''
        U =  qtp.Qobj(self.param_syst['U'], dims = [self.nb_levels, self.nb_levels])
        res = (U * state_init * U.dag() )
        return res#/np.trace(res.full())
        
        
    def _gate_from_simu(self, state_init): 
        '''param should contain 'simu' that runs simu from a dict with parameters and init  ;
        simu should onlytake init as Qobj'''
        if self._definition_type == '2-qubit simu':
            res =  self.param_syst['simu'](state_init, self.param_syst)  
            
        elif self._definition_type == '2system':
            res =  self.param_syst['simu'](state_init)
        return res#/np.trace(res.full())
            
    def gate(self, init):
        
        if self.store_outputs and isinstance(init, list):
            key = "output"
            if self.definition_type == "2system":
                key += str(hash(self.carac))
            key += str(np.sort(init))
                
            if key in self.param_syst.keys(): #if already processed
                # print("Used stored for output")
                return self.param_syst[key]
            
        if isinstance(init, list): #definition via index or label and prefactors
            ket_init = qtp.Qobj(np.zeros((self.d, 1)), dims =  [self.nb_levels, [1,1]])
            for tpl in init:
                if isinstance(tpl[1], int):#I need kets for superpositions
                    ket_init += tpl[0] * self._ket_index(tpl[1])
                    
                elif isinstance(tpl[1], Iterable) :
                    assert len(tpl[1]) == len(self.nb_levels)
                    for k in tpl[1]:
                        assert isinstance(k, int)
                        
                    ket_init += tpl[0] * self._ket_label(tpl[1])
                    
                else:
                    print("Error : init not recognized")
                    return None
                    
            state_init = (ket_init * ket_init.dag())
            
        elif isinstance(init, qtp.qobj.Qobj): #Qobj directly (necessary at some point)
            if init.type == 'ket':
                state_init = init * init.dag()
            elif init.type == 'oper':
                state_init = init
            else:
                print("Type of the initial state not recognized, if entered as qobj it should be 'oper' or 'ket'")
            
        else:
            print("Type of the initial state not recognized, should be list or qobj")
            
        state_init = state_init.unit() #it's a choice   
        
        
        #now the different possible cases
        if self._definition_type == 'kraus':
            res = self._gate_from_Kraus(state_init)
            
        elif self._definition_type == 'U':
            res = self._gate_from_U(state_init)
        
        elif self._definition_type == '2-qubit simu':
            res = self._gate_from_simu(state_init)
            
        elif self._definition_type == '2system':
            res = self._gate_from_simu(state_init)
        
        else :
            raise("Error ! \nDefinition type not recognized. \nPossible values are : \
                    'kraus', 'U', '2-qubit simu', '2system' ")
                    
        if self.store_outputs and isinstance(init, list):
            self.param_syst[key] = res
        self.nb_gate_call +=1 
        return res
            
            

 
## Actual tomography
# the studied function is gate that uses the definition type specifyed in self._definition_type

#fct_to_lambda
    def fct_to_lambda(self, in_labels = "comp_states", out_labels = "comp_states", draw_lambda = False, as_qobj = False):
        ''' in_labels or out_labels are at "comp_states" if only 00, 01, 10, 11 ; 
            "all" if all
            else : in a list of tuples'''
            
        #clean label lists
        if in_labels == "comp_states":
            in_labels = [(0,0), (0,1), (1,0), (1,1)]
        if out_labels == "comp_states":
            out_labels = [(0,0), (0,1), (1,0), (1,1)]
            
        if in_labels == "all":
            in_labels = [self._index_to_label(k) for k in range(self.d)]
        if out_labels == "all":
            out_labels = [self._index_to_label(k) for k in range(self.d)]
            
        assert isinstance(in_labels, list)
        assert isinstance(out_labels, list)
        
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
        if self.definition_type == "2system":
            key += str(hash(self.carac))
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
            n_i, m_i = _n_th([len(in_ind),len(in_ind)], i)
            
            rho_prime_i = rho_prime(in_ind[n_i], in_ind[m_i])
            
                
            for j in range(len(out_ind)**2):
                #we range over the pairs of out_labels
                n_j, m_j = _n_th([len(out_ind),len(out_ind)], j)
                
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
            

    def lambda_as_function(self, in_ind, out_ind):
        
        #saving trick
        key = "lambda_coef"
        if self.definition_type == "2system":
            key += str(hash(self.carac))
        key += str(in_ind)
        key += str(out_ind)
        
        if key in self.param_syst.keys(): #if already processed
            # print("Used stored for lambda_coef")
            return self.param_syst[key]
        
        n_in, m_in = _n_th([self.d, self.d], in_ind)
        n_out, m_out = _n_th([self.d, self.d], out_ind)
        
        
        if n_in == m_in:
            output = self.gate([[1, n_in]])
        else :
            n_n_prime = self.gate([[1, n_in]])
            m_m_prime = self.gate([[1, m_in]])
            plus_plus_prime = self.gate([[1/np.sqrt(2), n_in], [1/np.sqrt(2), m_in]])
            minus_minus_prime = self.gate([[1/np.sqrt(2), n_in], [1j/np.sqrt(2), m_in]])
            
            output = plus_plus_prime + 1j*minus_minus_prime - (1+1j)/2 * n_n_prime - (1+1j)/2 * m_m_prime
    
       
        coef = np.trace( output.full().dot(self._rho_nm(n_out, m_out).dag().full()) )
        
        self._param_syst[key] = coef
        return coef
        
        
        
        
## Leakage and Seepage 

    def L1(self, U_ideal = None, correc = None, labels_chi_1 = "comp_states"):
        '''default as 'comp_states' ie 00, 01, 10, 11'''
        
         
        if labels_chi_1 == "comp_states":
            labels_chi_1 = [(0,0), (0,1), (1,0), (1,1)]
            
        #trick to avoid redundant calculations
        key = "L1"
        if self.definition_type == "2system":
            key += str(hash(self.carac))
        key += str(hash(str(U_ideal)))
        key += str(hash(correc))
        key += str(np.sort(labels_chi_1))

        if key in self.param_syst.keys(): #if already processed
            # print("Used stored for L1")
            return self.param_syst[key]
        
            
            
        state_init = qtp.Qobj(np.zeros((self.d, self.d)), dims = [self.nb_levels, self.nb_levels])
        for dm in [self._dm_label(label)/len(labels_chi_1) for label in labels_chi_1]:
            state_init += dm
        
        if U_ideal is None:
            res = self.gate(state_init)
        
        else :
            assert U_ideal.shape == (self.d, self.d)
            proj1_dm_prime = self.gate(state_init)
            
            U_correction = correc(self, in_out_labels = labels_chi_1)
            assert U_correction.shape == (self.d, self.d)
            U_ideal_corrected = U_correction.conj().T.dot(U_ideal)
            env_U_ideal_corrected_dag = TomoEnv(system = None,
                                                definition_type = "U",
                                                nb_levels = self.nb_levels, 
                                                param_syst = { 'U' : U_ideal_corrected.conj().T}, 
                                                table_states = self._table_states)
                                                            
            res = env_U_ideal_corrected_dag.gate(proj1_dm_prime)
            res = res/np.trace(res.full()) #proved useful
        
        L1 = 1 - np.sum([np.trace((self._dm_label(label) * res).full()) for label in labels_chi_1])
        
        #once processed, we save it for later (part of trick from above)
        self._param_syst[key] = L1

            
        return L1
        

    def L2(self,  U_ideal = None, correc = None, labels_chi_1 = "comp_states"):
        
        if labels_chi_1 == "comp_states":
            labels_chi_1 = [(0,0), (0,1), (1,0), (1,1)]
        ind_chi_1 = [self._label_to_index(label) for label in labels_chi_1]
        ind_chi_2 = []
        for k in range(self.d):
            if not k in ind_chi_1:
                ind_chi_2.append(k)
                
        #trick to avoid redundant calculations
        key = "L2"
        if self.definition_type == "2system":
            key += str(hash(self.carac))
        key += str(hash(str(U_ideal)))
        key += str(hash(correc))
        key += str(np.sort(labels_chi_1))
        
        if key in self.param_syst.keys(): #if already processed
            # print("Used stored for L2")
            return self.param_syst[key]
            
        state_init = qtp.Qobj(np.zeros((self.d, self.d)), dims = [self.nb_levels, self.nb_levels])
        for dm in [self._dm_index(ind)/len(ind_chi_2) for ind in ind_chi_2]:
            state_init += dm
            
            
        if U_ideal is None:
            res = self.gate(state_init)
        
        else :
            assert U_ideal.shape == (self.d, self.d)
            proj2_dm_prime = self.gate(state_init)
            
            U_correction = correc(self, in_out_labels = labels_chi_1)
            assert U_correction.shape == (self.d, self.d)
            U_ideal_corrected = U_correction.conj().T.dot(U_ideal)
            env_U_ideal_corrected_dag = TomoEnv(system = None,
                                                definition_type = "U",
                                                nb_levels = self.nb_levels, 
                                                param_syst = { 'U' : U_ideal_corrected.conj().T}, 
                                                table_states = self._table_states)
                                                                    
            res = env_U_ideal_corrected_dag.gate(proj2_dm_prime)
            res = res/np.trace(res.full()) #proved useful
            
        L2 = np.sum([np.trace((self._dm_label(label) * res).full()) for label in labels_chi_1]) 
        
        #once processed, we save it for later (part of trick from above)
        self._param_syst[key] = L2

        return L2
    

###Not needed
#fct_to_PTM
    def fct_to_PTM(self, draw_PTM = False, as_qobj = False):
                        
        
        #set the basis
        E_tilde_basis = self._basis_E_tilde_pauli()
        E_tilde_prime = [self.gate(E_tilde) for E_tilde in E_tilde_basis]
        
        d = self.d
        
        PTM_mat = np.zeros((d**2, d**2))*1j
        
        for i in range(d**2):
            for j in range(d**2):
                PTM_mat[i,j] = 1/d * np.trace(E_tilde_basis[i].full().dot(E_tilde_prime[j].full()))
                
        if draw_PTM:
            draw_mat(PTM_mat, "PTM") 
            
            
        if as_qobj:
            return qtp.Qobj(inpt = PTM_mat, dims = [E_tilde_basis[0].dims, E_tilde_basis[0].dims]) #?
        else:
            return PTM_mat
        
        
#lambda_to_chi
    def lambda_to_chi(self, lambda_mat, draw_chi = False, as_qobj = False):
    
        basis_E_tilde = self._basis_E_tilde_pauli()
        
        lambda_qobj = qtp.Qobj(lambda_mat)
        chi_mat = qtp.qpt(lambda_qobj, [basis_E_tilde])
        
        if draw_chi:
            draw_mat(chi_mat, "\chi")
    
        if as_qobj:
            return qtp.Qobj(inpt = chi_mat, dims = [basis_E_tilde[0].dims, basis_E_tilde[0].dims]) #?
        else:
            return chi_mat

    
#chi_to_kraus
    def chi_to_kraus(self, chi_mat, draw_kraus = False, as_qobj = False):
        
        #set the basis
        basis_E_tilde = self._basis_E_tilde_pauli()
        d = self.d
        nb_levels = self.nb_levels
        
        D, U = la.eigh(chi_mat)
        U = U
        
        res = []
        for i in range(len(D)):
            if np.abs(D[i]) > 10**(-9): #we discard null eigenvalues
                res.append(qtp.Qobj(np.zeros((d,d)), dims = [nb_levels, nb_levels]))
                for j in range(len(basis_E_tilde)):
                        res[-1]+= np.sqrt(D[i])* U[j,i] * basis_E_tilde[j]
                # res[-1].unit()        
        
        if draw_kraus:
            if len(res) == 1:
                draw_mat(res[0].full(), "E^0")
            else :
                draw_mat_mult([op.full() for op in res],
                            ["E^{"+str(i)+"}" for i in range(len(res))])
                    
        if as_qobj:
            return res 
        else:
            return [op.full() for op in res]    
            
#chi_to_PTM
    def chi_to_PTM(self, chi_mat, draw_PTM = False, as_qobj = False):
        #set the basis
        basis_E_tilde = self._basis_E_tilde_pauli()
        d = self.d
        
        PTM_mat = np.zeros((d**2, d**2))*1j
        
        for i in range(d**2):
            for j in range(d**2):
                for k in range(d**2):
                    for l in range(d**2):
                        PTM_mat[i,j] += 1/d * chi_mat[k,l] * \
                        np.trace(basis_E_tilde[i].full().dot(
                                    basis_E_tilde[k].full()).dot(
                                            basis_E_tilde[j].full()).dot(
                                                basis_E_tilde[l].full())
                                )
                        
                        
        if draw_PTM:
            draw_mat(PTM_mat, "PTM") 
            
        if as_qobj:
            return qtp.Qobj(inpt = PTM_mat, dims = [basis_E_tilde[0].dims, basis_E_tilde[0].dims]) #?
        else:
            return PTM_mat
            
            
## Summaries

    def fct_to_chi(self, draw_lambda = False, draw_chi = False, as_qobj = False):
                        
        lambda_mat = self.fct_to_lambda(draw_lambda = draw_lambda)
        chi_mat = self.lambda_to_chi(lambda_mat, draw_chi = draw_chi, as_qobj = as_qobj)
        return chi_mat
    
    def fct_to_kraus(self, draw_lambda = False, draw_chi = False, draw_kraus = False, as_qobj = False):
                        
                        
        lambda_mat = self.fct_to_lambda(draw_lambda=draw_lambda)
        chi_mat = self.lambda_to_chi(lambda_mat, draw_chi=draw_chi)
        kraus_list = self.chi_to_kraus(chi_mat, draw_kraus=draw_kraus, as_qobj = qs_qobj)
        return kraus_list
        
## Through beta (mainly to check coherence in lambda_<->_chi)


#define beta in functionnal and matricial form
    def _beta_4D(self, j,k,m,n):
        
        #set conventions
        E_tilde = self._E_tilde_pauli
        rho_flat = self._rho_j #to calculate each term
        
        
        return np.trace(E_tilde(m).full().dot(
                            rho_flat(j).full()).dot(
                            E_tilde(n).dag().full()).dot(
                                rho_flat(k).dag().full())
                    )
    
    
    def _beta_2D(self, mu, nu):
    
        d = self.d
        #translate indices
        j,k = _n_th([d**2, d**2], mu)
        m,n = _n_th([d**2, d**2], nu)
        
        return self._beta_4D(j,k,m,n)
            
    def _beta_mat_form(self):
        
        '''The beta matrix returned will be d^4 x d^4 with d the product of all terms in nb_levels'''  
                        
        d = self.d
        #build beta
        beta_mat = np.zeros((d**4, d**4))*1j
    
        for mu in range(d**4):
            for nu in range(d**4):
                beta_mat[mu, nu] = self._beta_2D(mu, nu)
                                    
        return beta_mat
        
        
    #define chi_to_lambda using beta
    def chi_to_lambda_with_beta(self, chi_mat, draw_lambda = False, as_qobj = False):
                            
        d = self.d
        
        lambda_th_mat = np.zeros((d**2,d**2))*1j
        for j in range(d**2):
            for k in range(d**2):
                for m in range(d**2):
                    for n in range(d**2):
                        lambda_th_mat[j,k] += self._beta_4D(j,k,m,n)*chi_mat[m,n]
                        
        if draw_lambda:
            draw_mat(lambda_th_mat, "\lambda^{th}") 
                        
        if as_qobj:
            return qtp.Qobj(inpt = lambda_th_mat, dims = [[self.nb_levels, self.nb_levels], [self.nb_levels, self.nb_levels]]) #?
        else:
            return lambda_th_mat
    
    #define lambda_to_chi using beta
    def lambda_to_chi_with_beta(self, lambda_mat, draw_chi = False, as_qobj = False):
        '''We use the beta matrix which is d^4 x d^4 and we pseudo-inverse it'''

        d = self.d

        #Start calculations
        beta_mat = self._beta_mat_form()
    
        lambda_vec = lambda_mat.reshape((d**4, 1))
        
        chi_vec = la.solve(beta_mat, lambda_vec)
        
        chi_th_mat = chi_vec.reshape((d**2, d**2))
        
        if draw_chi:
            draw_mat(chi_th_mat, "\chi^{th}") 
            
        if as_qobj:
            return qtp.Qobj(inpt = chi_th_mat, dims = [[self.nb_levels, self.nb_levels], [self.nb_levels, self.nb_levels]]) #?
        else:
            return chi_th_mat

        
### Tests


# nb_levels =  [4,3]
# U_id = qtp.qeye(nb_levels)
# 
# param_test = {
#     'U' : U_id
# }
# 
# 
# env = TomoEnv(system = None,
#                 definition_type = "U",
#                 nb_levels = nb_levels, 
#                 param_syst = param_test, 
#                 table_states = None)
# 
# deb = time.time()
# lambda_mat = env.fct_to_lambda(draw_lambda = False, as_qobj = False)
# print("It took ", time.time() - deb, "seconds")
# 
# print(env.L1().real)
# print(env.L2().real)
# 
# plt.show()


# 
# deb = time.time()
# chi_mat = env.lambda_to_chi(lambda_mat, draw_chi = True, as_qobj = False)
# print("It took ", time.time() - deb, "seconds")
# 
# 
# # deb = time.time()
# # chi_th_mat = env.lambda_to_chi_with_beta(lambda_mat, draw_chi = True, as_qobj = False)
# # print("It took ", time.time() - deb, "seconds")
# #     
