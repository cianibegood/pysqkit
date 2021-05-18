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
    ''' **kwargs should contain at least 'op_list' a list of Kraus operators'''
    
    op_list = kwargs['op_list']
    
    for op in op_list:
        if state_init.shape != op.shape:
            print("The dimensions don't match")
            return None
        
    if state_init.type == "_ket": 
        print("Initial state has been transformed into a density matrix")
        state_init = state_init * state_init.dag()
    
    res = qtp.Qobj(np.zeros(state_init.shape), dims = state_init.dims)
    for op in op_list:
        assert isinstance(op, qtp.Qobj)
        res+= op*state_init*op.dag()
    
    return res


def _gate_from_U(state_init, **kwargs): 
    '''kwargs should contain U which is a 2D numpy array'''

    U = kwargs['U']
    
    if state_init.shape[0] != U.shape[0] :
        print("The dimensions don't match")
        return None
    
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

## get rid of dictionnaries (standardize functions)
def rid_of_dict(fct, dict):
    '''dict should contain all the necessary parameters for fct
    
    fct should make sure that the dimensions match'''
    
    def res(init_state):
        return fct(init_state, **dict)
        
    return res

## Function to lambda
def fct_to_lambda(fct, 
                    nb_levels: Union[int, Iterable[int]],
                    base_rho="nm", 
                    draw_lambda = False):
    ''' 'fct' takes an initial state and returns the output state
    
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
            return fct(rho(n, m, nb_levels))
        else :
            n_n = rho(n, n, nb_levels)
            m_m = rho(m, m, nb_levels)
            
            plus = (_ket(n, nb_levels) + _ket(m, nb_levels)).unit()
            plus_plus = plus * plus.dag()
            
            minus = (_ket(n, nb_levels) +  1j*_ket(m, nb_levels)).unit()
            minus_minus = minus * minus.dag()

            n_n_prime = fct(n_n)
            m_m_prime = fct(m_m)
            plus_plus_prime = fct(plus_plus)
            minus_minus_prime = fct(minus_minus)

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
    
##Function_to_PTM
def fct_to_PTM(fct, 
                    nb_levels: Union[int, Iterable[int]],
                    draw_PTM = False):
    ''' 'fct' takes an initial state and returns the output state
    
        'nb_levels' is an int or a list of the number of levels of each system
        
        fct should be able to take the operators of the pauli basis in input'''
                    
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    #set the basis
    E_tilde_basis = _basis_E_tilde_pauli(nb_levels)
    E_tilde_prime = [fct(E_tilde) for E_tilde in E_tilde_basis]
    
    PTM_mat = np.zeros((d**2, d**2))*1j
    
    for i in range(d**2):
        for j in range(d**2):
            PTM_mat[i,j] = 1/d * np.trace(E_tilde_basis[i].full().dot(E_tilde_prime[j].full()))
            
    if draw_PTM:
        draw_mat(PTM_mat, "PTM") 
        
    return PTM_mat
    
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
            
## chi_to_PTM
def chi_to_PTM(chi_mat, 
                nb_levels: Union[int, Iterable[int]],
                draw_PTM = False):
    
    '''chi_mat must have been calculated using the pauli_like operator base'''
    
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    #set the basis
    basis_E_tilde = _basis_E_tilde_pauli(nb_levels)
    
    
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
        
    return PTM_mat



## Summary Kraus_to_fct
def kraus_to_fct(kraus_list):
    '''returns function that only takes an init state as argument'''
    return rid_of_dict(_gate_from_Kraus, {'op_list' : kraus_list})
    
## Summary U_to_fct
def U_to_fct(U):
    '''returns function that only takes an init state as argument'''
    return rid_of_dict(_gate_from_U, {'U': U})

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
    
    
## Summary kraus_to_chi (for verifications mainly)
def kraus_to_chi(kraus_list, 
                 nb_levels: Union[int, Iterable[int]],
                 base_rho="nm", 
                 base_E_tilde="Pauli gen", 
                 draw_chi = False):
    
    return fct_to_chi(_gate_from_Kraus, nb_levels = nb_levels, draw_chi = draw_chi, **{'op_list' : kraus_list})
    

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
    
##beta (mainly to check coherence in lambda_<->_chi)
#define beta in functionnal and matricial form
def _beta_4D(j,k,m,n,
           nb_levels: Union[int, Iterable[int]],
           base_rho="nm", 
           base_E_tilde="Pauli gen" ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    
    #set conventions
    if base_E_tilde != "Pauli gen":
        print("This base_E_tilde is non treated, instead took default basis of Pauli-like operators")
    E_tilde = _E_tilde_pauli
    
    if base_rho != "nm":
        print("This base_rho is non treated, instead took default basis of |n><m|")
    rho_flat = _rho_nm_flat #to calculate each term
    
    
    return np.trace(E_tilde(m, nb_levels).full().dot(
                        rho_flat(j, nb_levels).full()).dot(
                         E_tilde(n, nb_levels).dag().full()).dot(
                              rho_flat(k, nb_levels).dag().full())
                   )


def _beta_2D(mu, nu, 
           nb_levels: Union[int, Iterable[int]],
           base_rho="nm", 
           base_E_tilde="Pauli gen"):
    
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)

    #translate indices
    j,k = _n_th([d**2, d**2], mu)
    m,n = _n_th([d**2, d**2], nu)
    
    return _beta_4D(j,k,m,n,
           nb_levels,
           base_rho, 
           base_E_tilde)
           
def _beta_mat_form(nb_levels: Union[int, Iterable[int]],
                    base_rho="nm", 
                    base_E_tilde="Pauli gen"):
    '''nb_levels is an int or a list of the number of levels of each system
    
    The beta matrix returned will be d^4 x d^4 with d the product of all terms in nb_levels'''  
                     
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    #we fix that first
    if base_E_tilde != "Pauli gen":
        print("This base_E_tilde is non treated, instead took default basis of Pauli-like operators")
    E_tilde = _E_tilde_pauli
    
    if base_rho != "nm":
        print("This base_rho is non treated, instead took default basis of |n><m|")
    rho_flat = _rho_nm_flat #to calculate each term
    
    #build beta
    beta_mat = np.zeros((d**4, d**4))*1j

    for mu in range(d**4):
        for nu in range(d**4):
            beta_mat[mu, nu] = _beta_2D(mu, nu,
                                nb_levels,
                                base_rho, 
                                base_E_tilde)
                                
    return beta_mat
    
    
#define chi_to_lambda using beta
def chi_to_lambda_beta(chi_mat, 
                       nb_levels: Union[int, Iterable[int]],
                       base_rho="nm",  
                       base_E_tilde="Pauli gen", 
                       draw_lambda = False):
                        
    ''' 'nb_levels' is an int or a list of the number of levels of each system'''
                
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    #we fix eventual issues here
    if base_E_tilde != "Pauli gen":
        print("This base_E_tilde is non treated, instead took default basis of Pauli-like operators")
        base_E_tilde  = "Pauli_gen"
    
    if base_rho != "nm":
        print("This base_rho is non treated, instead took default basis of |n><m|")
        base_rho = "nm"
    
    lambda_th_mat = np.zeros((d**2,d**2))*1j
    for j in range(d**2):
        for k in range(d**2):
            for m in range(d**2):
                for n in range(d**2):
                    lambda_th_mat[j,k] += _beta_4D(j,k,m,n, 
                                            nb_levels, base_rho, base_E_tilde)*chi_mat[m,n]
                    
    if draw_lambda:
        draw_mat(lambda_th_mat, "\lambda^{th}") 
                    
    return lambda_th_mat

#define lambda_to_chi using beta
def lambda_to_chi_beta(lambda_mat, 
                       nb_levels: Union[int, Iterable[int]],
                       base_rho="nm",  
                       base_E_tilde="Pauli gen", 
                       draw_chi = False):
    ''' 'nb_levels' is an int or a list of the number of levels of each system
    
    We use the beta matrix which is d^4 x d^4 and we pseudo-inverse it'''
                
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)

    #we fix eventual issues here
    if base_E_tilde != "Pauli gen":
        print("This base_E_tilde is non treated, instead took default basis of Pauli-like operators")
        base_E_tilde  = "Pauli_gen"
    
    if base_rho != "nm":
        print("This base_rho is non treated, instead took default basis of |n><m|")
        base_rho = "nm"

    #Start calculations
    beta_mat = _beta_mat_form(nb_levels, base_rho, base_E_tilde)
    
    lambda_vec = lambda_mat.reshape((d**4, 1))
    
    chi_vec = la.solve(beta_mat, lambda_vec)
    
    chi_th_mat = chi_vec.reshape((d**2, d**2))
    
    if draw_chi:
        draw_mat(chi_th_mat, "\chi^{th}") 
        
    return chi_th_mat
    
    
### Tests
# mult = 2
# lvl_each = 2
# U_mult = qtp.Qobj(2*(np.random.rand(lvl_each**mult, lvl_each**mult)-.5) + \
#                 2j*(np.random.rand(lvl_each**mult, lvl_each**mult)-.5), 
#                    dims = [[lvl_each]*mult, [lvl_each]*mult]).unit() #qtp.qeye([2]*mult)
#                    
#                    
# U = U_mult
# nb_levels = [lvl_each, lvl_each]
# 
# 
# param_test = {
#     'U' : U
# }
# 
# 
# deb = time.time()
# lambda_mat = fct_to_lambda(_gate_from_U, nb_levels, draw_lambda = True, **param_test)
# print("It took ", time.time() - deb, "seconds")
# 
# 
# deb = time.time()
# chi_mat = lambda_to_chi(lambda_mat, nb_levels, draw_chi = True)
# print("It took ", time.time() - deb, "seconds")
# 
# 
# deb = time.time()
# chi_th_mat = lambda_to_chi_beta(lambda_mat, nb_levels, draw_chi = True)
# print("It took ", time.time() - deb, "seconds")
    
