from typing import Tuple, Optional, Dict, Iterable, Union, Callable, List
import os
import sys

from pathlib import Path#not necessary ?
parent_path = ".."  #str(Path(__file__).resolve().parents[1])
sys.path.append(parent_path)

import time
import datetime

import qutip as qtp 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg as la #not necessary ?

import pysqkit #not necessary ?


from pysqkit.solvers.solvkit import integrate
from mpl_toolkits.axes_grid1 import make_axes_locatable

### General Tools

##Abstacts
def _n_th(maxs, n):  
    '''returns n-th tuple with the i_th term varying between 0 and maxs[i]'''
    temp = np.zeros(len(maxs))
    for i in range(0, len(temp)):
        temp[i] = (n//np.prod(maxs[i+1:]))%np.prod(maxs[i])
    
    res = [int(k) for k in temp]
    return res

## Defining gates
def _gate_from_Kraus(state_init, **kwargs):
    ''' **kwargs should contain at least 'op_list' a list of Kraus operators '''
    
    if state_init.type == "_ket": 
        print("Initial state has been transformed into a density matrix")
        state_init = state_initnit * state_init.dag()
    
    res = qtp.Qobj(np.zeros(state_init.shape), dims = state_init.dims)
    for op in kwargs['op_list']:
        assert isinstance(op, qtp.Qobj)
        res+= op*state_init*op.dag()
    
    return res


def _gate_from_U(state_init, **kwargs): 
    ''' **kwargs should contain at least 'U' the unitary operator describing the gate '''
    U = kwargs['U']
    
    if state_init.type == "_ket": 
        return qtp.Qobj(U) * state_init
    elif state_init.type == "oper":
        return qtp.Qobj(U) * state_init * qtp.Qobj(U).dag()
    else :
        print("Type of state_init not recognized, must be _ket or oper")
        
    
## Visualisation
def draw_mat(mat, mat_name, vmin = np.NaN, vmax = np.NaN):
    ''' To represent a matrix, we show its module, real part and imaginary part'''
    
    if np.isnan(vmin):
        vmin = min(np.min(mat.real), np.min(mat.imag))
    if np.isnan(vmax):
        vmax = np.max([np.max(mat.real), np.max(mat.imag)])
    fig, ax = plt.subplots(1, 3, figsize = (12,5))

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
    
    fig.show()
    #show and save ??
    
    
def draw_mat_mult(mat_list, mat_name_list, vmin = np.NaN, vmax = np.NaN):
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
    
    fig.show()
    
    
### Tomography Preparation

##State base 
def _ket(n: int, 
        nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    return  qtp.fock(nb_levels, _n_th(nb_levels, n))

def _bra(m,
        nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    return  qtp.fock(nb_levels, _n_th(nb_levels, m)).dag()   
     
def _rho_nm(n, m,
           nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    return _ket(n, nb_levels)*_bra(m, nb_levels)

def _rho_nm_flat(j, 
                nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    n, m = _n_th([d,d], j)
    
    return _rho_nm(n,m, nb_levels)
    

def _basis_rho_nm(nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    res = []
    for k in range(d**2):
        res.append(_rho_nm_flat(k , nb_levels))

    return res 
    
    
    
## Base of Operators
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
            h_k_lvl_inf = _Pauli_gen(a1*(local_d-1) + a2, local_d-1)  #we take one at same coordinates in lower level basis
            res[:-1, :-1] = h_k_lvl_inf.full()
            return qtp.Qobj(res)

        else : #a1 = a2 = lvls-1
            res = np.zeros((local_d, local_d))*0j
            res[:-1, :-1] = qtp.qeye(local_d-1).full()
            res[-1, -1] = 1-local_d
            res = np.sqrt(2/(local_d*(local_d-1)))*res
            return qtp.Qobj(res)
        
    
def _E_tilde_pauli(i, nb_levels: Union[int, Iterable[int]]):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
        
    pauli_list = []
    
    pauli_maxs = [int(max_lvl**2) for max_lvl in nb_levels]
    
    tpl = _n_th(pauli_maxs, i) #tuple of indices of the P_i that will appear in the product defining E_tilde 
                              #(as in sigma_i x sigma_j x sigma_k)
                              #For L levels, there are L^2 _Pauli_gen matrices
    for j in range(len(tpl)):
        ind = tpl[j]
        pauli_list.append(_Pauli_gen(ind, nb_levels[j]))
        
    return qtp.tensor(pauli_list)
                           
def _basis_E_tilde_pauli(nb_levels: Union[int, Iterable[int]]):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    res = []
    for i in range(d**2):
        res.append(_E_tilde_pauli(i , nb_levels))

    return res
    
    
### Actual Tomography

## Function to lambda
def fct_to_lambda(fct, 
                    nb_levels: Union[int, Iterable[int]],
                    base_rho="nm", 
                    draw_lambda = False, 
                    **kwargs):
    ''' 'fct' takes an initial state and kwargs and returns the output state
    
        'nb_levels' is an int or a list of the number of levels of each system'''
                    
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    #set the basis
    if base_rho != "nm":
        print("This base_rho is non treated, instead took default basis of |n><m|")
    rho = _rho_nm #functions to calculate each term
            
            
    #function to calculate rho_primes
    def rho_prime(n,m):
        if n == m:
            return fct(rho(n, m, nb_levels), **kwargs)
        else :
            n_n = rho(n, n, nb_levels)
            m_m = rho(m, m, nb_levels)
            
            plus = (_ket(n, nb_levels) + _ket(m, nb_levels)).unit()
            plus_plus = plus * plus.dag()
            
            minus = (_ket(n, nb_levels) +  1j*_ket(m, nb_levels)).unit()
            minus_minus = minus * minus.dag()

            n_n_prime = fct(n_n, **kwargs)
            m_m_prime = fct(m_m, **kwargs)
            plus_plus_prime = fct(plus_plus, **kwargs)
            minus_minus_prime = fct(minus_minus, **kwargs)

            return plus_plus_prime + 1j*minus_minus_prime - (1+1j)/2 * n_n_prime - (1+1j)/2 * m_m_prime

            
    #skeleton
    lambda_mat = np.zeros((d**2, d**2))*1j

    #filling
    for i in range(d**2):
        n_i, m_i = _n_th([d,d], i)
        
        rho_prime_i = rho_prime(n_i, m_i)
            
            
        for j in range(d**2):
            n_j, m_j = _n_th([d,d], j)
            
            lambda_mat[i,j] = np.trace(rho_prime_i.full().dot(
                                             rho(n_j, m_j, nb_levels).dag().full())  
                                       )
            
    if draw_lambda:
        draw_mat(lambda_mat, "\lambda")  
        
    return lambda_mat
    
## lambda_to_chi  
def lambda_to_chi(lambda_mat, 
                    nb_levels: Union[int, Iterable[int]], 
                    base_E_tilde="Pauli gen", 
                    draw_chi = False):
                        
    ''' 'nb_levels' is an int or a list of the number of levels of each system'''
                    
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
                        
    #set the basis
    if base_E_tilde != "Pauli gen":
        print("This base_E_tilde is non treated, instead took default basis of Pauli-like operators")
    basis_E_tilde = _basis_E_tilde_pauli(nb_levels)
    
    lambda_qobj = qtp.Qobj(lambda_mat)
    chi_mat = qtp.qpt(lambda_qobj, [basis_E_tilde])
      
    if draw_chi:
        draw_mat(chi_mat, "\chi")

    return chi_mat

## chi_to_kraus
def chi_to_kraus(chi_mat, 
                 nb_levels: Union[int, Iterable[int]], 
                 base_E_tilde="Pauli gen", 
                 draw_kraus = False):
    
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    #set the basis
    if base_E_tilde != "Pauli gen":
        print("This base_E_tilde is non treated, instead took default basis of Pauli-like operators")
    basis_E_tilde = _basis_E_tilde_pauli(nb_levels)
    
    D, U = la.eigh(chi_mat)
    U = U
    
    res = []
    for i in range(len(D)):
        if np.abs(D[i]) > 10**(-9): #we discard null eigenvalues
            res.append(qtp.Qobj(np.zeros((d,d)), dims = [nb_levels, nb_levels]))
            for j in range(len(basis_E_tilde)):
                    res[-1]+= np.sqrt(D[i])* U[j,i] * basis_E_tilde[j]
                    
    
    if draw_kraus:
        if len(res) == 1:
            draw_mat(res[0].full(), "E^0")
        else :
            draw_mat_mult([op.full() for op in res],
                          ["E^{"+str(i)+"}" for i in range(len(res))])
                   
    return res
            
            
##kraus_to_chi (for verifications mainly)
def kraus_to_chi(kraus_list, 
                 nb_levels: Union[int, Iterable[int]],
                 base_rho="nm", 
                 base_E_tilde="Pauli gen", 
                 draw_chi = False):
    
    return fct_to_chi(_gate_from_Kraus, nb_levels = nb_levels, draw_chi = draw_chi, **{'op_list' : kraus_list})
    
## Summary fct_to_chi
def fct_to_chi(fct, 
                nb_levels: Union[int, Iterable[int]],
                base_rho="nm", 
                base_E_tilde="Pauli gen", 
                draw_lambda = False, 
                draw_chi = False, 
                **kwargs):
                    
                    
    lambda_mat = fct_to_lambda(fct, nb_levels, 
                    base_rho=base_rho, draw_lambda=draw_lambda, **kwargs)
    
    chi_mat = lambda_to_chi(lambda_mat, nb_levels,
                    base_E_tilde=base_E_tilde, draw_chi=draw_chi)

    return chi_mat

## Summary fct_to_kraus
def fct_to_kraus(fct, 
                nb_levels: Union[int, Iterable[int]], 
                base_rho="nm", 
                base_E_tilde="Pauli gen", 
                draw_lambda = False, 
                draw_chi = False,
                draw_kraus = False,
                **kwargs):
                    
                    
    lambda_mat = fct_to_lambda(fct, nb_levels, 
                    base_rho=base_rho, draw_lambda=draw_lambda, **kwargs)
    
    chi_mat = lambda_to_chi(lambda_mat, nb_levels,
                    base_E_tilde=base_E_tilde, draw_chi=draw_chi)
    
    kraus_list = chi_to_kraus(chi_mat, nb_levels, 
                    base_E_tilde=base_E_tilde, draw_kraus=draw_kraus)

    return kraus_list

    
