from typing import Tuple, Optional, Dict, Iterable, Union, Callable, List
import os
import sys

import time
import datetime

import qutip as qtp 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg as la

# import pysqkit
# from pysqkit.solvers.solvkit import integrate

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
    
##  Tomo env class 
    
    
class TomoEnv:   
    def __init__(
        self,
        nb_levels: Union[int, Iterable[int]],
        definition_type: str,
        param_syst,
        table_states = None):
            '''  table_states is None if we want to take the bare basis'''
            
            self._definition_type = definition_type
            
            if isinstance(nb_levels, int):
                nb_levels = [nb_levels]
            self._nb_levels = nb_levels
            self._d = np.prod(nb_levels)
            
            if table_states is None:
                self._table_states = []
                for k in range(self._d):
                    self._table_states.append(qtp.fock(self._nb_levels, _n_th(self._nb_levels, k) ))
            else :
                self._table_states = table_states #should be a list of states in ket form, ordered by ascending label
            
            
            if self._table_states[0].type == 'ket':
                self._state_representation = 'ket'

            else: 
                print("Error : the type of the states in table_stats is not recognized, it should be ket")
                raise ValueError
                
            self._param_syst = param_syst #should have a function called simu that takes dict and init
            #init can be a list to use the state_table OR a a Qobj
        
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
            res = 0
            for k in range(len(label)):
                res+= label[k]*nb_levels[k]
            return res
            


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
    def _gate_from_Kraus(self, init):
        ''' init should be a list of tuples shaped like this :
        tuple[0] is the prefactor and tuple[1] is the label(if size >1) or the index(if size 1)  ;
        param should contain a 'op_list' list of Kraus operators'''
        
        param = self.param_syst
        op_list = param['op_list']
        
        if isinstance(init, list): #definition via index or label and prefactors
            ket_init = qtp.Qobj(np.zeros((self.d, 1)), dims =  [self.nb_levels, [1,1]])
            for tpl in init:
                if isinstance(tpl[1], int):#I need kets for superpositions
                    ket_init += tpl[0] * self._ket_index(tpl[1])
                    
                elif isinstance(tpl[1], Iterable[int]):
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
        
        
        state_init = state_init.unit() #it's a choice
        
        res = qtp.Qobj(np.zeros(state_init.shape), dims = state_init.dims)
        for op in op_list:
            assert isinstance(op, qtp.Qobj)
            res+= op*state_init*op.dag()
        
        return res
    
    
    def _gate_from_U(self, init): 
        '''param should contain U which is a 2D numpy array'''
    
        param = self.param_syst
        U = param['U']
        
        if isinstance(init, list): #definition via index or label and prefactors
            ket_init = qtp.Qobj(np.zeros((self.d, 1)), dims =  [self.nb_levels, [1,1]])
            for tpl in init:
                if isinstance(tpl[1], int):#I need kets for superpositions
                    ket_init += tpl[0] * self._ket_index(tpl[1])
                    
                elif isinstance(tpl[1], Iterable[int]):
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
            
        state_init = state_init.unit() #it's a choice   
        
        return qtp.Qobj(U) * state_init * qtp.Qobj(U).dag()
        
        
    def _gate_from_simu(self, init):
        '''param should contain 'simu' that runs simu from a dict with parameters and init  ;
        simu should be able to deal with a difinition via list of directly a Qobj'''
        return self.param_syst['simu'](init, self.param_syst)  #warning, for superpositions, I need kets
        #normalize every state that goes in ?
            
    def gate(self, init):
        if self._definition_type == 'kraus':
            return self._gate_from_Kraus(init)
            
        elif self._definition_type == 'U':
            return self._gate_from_U(init)
        
        elif self._definition_type == 'simu':
            return self._gate_from_simu(init)
        
        else :
            print("Error ! \nDefinition type not recognized. \nPossible values are : \
                    'kraus', 'U', 'simu' ")
            

 
## Actual tomography
# the studied function is gate that uses the definition type specifyed in self._definition_type

#fct_to_lambda
    def fct_to_lambda(self, draw_lambda = False, as_qobj = False):
                
        #function to calculate rho_primes
        def rho_prime(n,m):
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
        lambda_mat = np.zeros((d**2, d**2))*1j
    
        #filling
        for i in range(d**2):
            n_i, m_i = _n_th([d,d], i)
            
            rho_prime_i = rho_prime(n_i, m_i)
            
                
            for j in range(d**2):
                n_j, m_j = _n_th([d,d], j)
                
                lambda_mat[i,j] = np.trace(rho_prime_i.full().dot(
                                                self._rho_nm(n_j, m_j).dag().full())  
                                        )
                
        if draw_lambda:
            draw_mat(lambda_mat, "\lambda")  
        
        if as_qobj:
            return qtp.Qobj(inpt = lambda_mat, dims = [rho_prime_i.dims, rho_prime_i.dims]) #?
        else:
            return lambda_mat

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


nb_levels =  [3, 2]
U_id = qtp.qeye(nb_levels)

param_test = {
    'U' : U_id
}


env = TomoEnv(nb_levels, "U", param_test)
# 
# 
# deb = time.time()
# lambda_mat = env.fct_to_lambda(draw_lambda = True, as_qobj = False)
# print("It took ", time.time() - deb, "seconds")

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
# plt.show()