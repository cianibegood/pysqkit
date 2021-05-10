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
def n_th(maxs, n):  
    '''returns n-th tuple with the i_th term varying between 0 and maxs[i]'''
    temp = np.zeros(len(maxs))
    for i in range(0, len(temp)):
        temp[i] = (n//np.prod(maxs[i+1:]))%np.prod(maxs[i])
    
    res = [int(k) for k in temp]
    return res
    
    
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
    
    plt.show()
    #show and save ??
    
### Tomography Preparation

##State base 
def ket(n: int, 
        nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    return  qtp.fock(nb_levels, n_th(nb_levels, n))

def bra(m,
        nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    return  qtp.fock(nb_levels, n_th(nb_levels, m)).dag()   
     
def rho_nm(n, m,
           nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    return ket(n, nb_levels)*bra(m, nb_levels)

def rho_nm_flat(j, 
                nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    n, m = n_th([d,d], j)
    
    return rho_nm(n,m, nb_levels)
    

def basis_rho_nm(nb_levels: Union[int, Iterable[int]] ):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    res = []
    for k in range(d**2):
        res.append(rho_nm_flat(k , nb_levels))

    return res 
    
    
    
## Base of Operators
def Pauli_gen(j, local_d):#operator P_j for 1 system
    
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
            h_k_lvl_inf = Pauli_gen(a1*(local_d-1) + a2, local_d-1)  #we take one at same coordinates in lower level basis
            res[:-1, :-1] = h_k_lvl_inf.full()
            return qtp.Qobj(res)

        else : #a1 = a2 = lvls-1
            res = np.zeros((local_d, local_d))*0j
            res[:-1, :-1] = qtp.qeye(local_d-1).full()
            res[-1, -1] = 1-local_d
            res = np.sqrt(2/(local_d*(local_d-1)))*res
            return qtp.Qobj(res)
        
    
def E_tilde_pauli(i, nb_levels: Union[int, Iterable[int]]):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
        
    pauli_list = []
    
    pauli_maxs = [int(max_lvl**2) for max_lvl in nb_levels]
    
    tpl = n_th(pauli_maxs, i) #tuple of indices of the P_i that will appear in the product defining E_tilde 
                              #(as in sigma_i x sigma_j x sigma_k)
                              #For L levels, there are L^2 pauli_gen matrices
    for j in range(len(tpl)):
        ind = tpl[j]
        pauli_list.append(Pauli_gen(ind, nb_levels[j]))
        
    return qtp.tensor(pauli_list)
                           
def basis_E_tilde_pauli(nb_levels: Union[int, Iterable[int]]):
    '''nb_levels is an int or a list of the number of levels of each system'''
    if isinstance(nb_levels, int):
        nb_levels = [nb_levels]
    d = np.prod(nb_levels)
    
    res = []
    for i in range(d**2):
        res.append(E_tilde_pauli(i , nb_levels))

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
    rho = rho_nm #functions to calculate each term
    rho_flat = rho_nm_flat
            
            
    #function to calculate rho_primes
    def rho_prime(n,m):
        if n == m:
            return fct(rho(n, m, nb_levels), **kwargs)
        else :
            n_n = rho(n, n, nb_levels)
            m_m = rho(m, m, nb_levels)
            
            plus = (ket(n, nb_levels) + ket(m, nb_levels)).unit()
            plus_plus = plus * plus.dag()
            
            minus = (ket(n, nb_levels) +  1j*ket(m, nb_levels)).unit()
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
        n_i, m_i = n_th([d,d], i)
        
        rho_prime_i = rho_prime(n_i, m_i)
            
            
        for j in range(d**2):
            n_j, m_j = n_th([d,d], j)
            
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
    basis_E_tilde = basis_E_tilde_pauli(nb_levels)
    
    lambda_qobj = qtp.Qobj(lambda_mat)
    chi_mat = qtp.qpt(lambda_qobj, [basis_E_tilde])
      
    if draw_chi:
        draw_mat(chi_mat, "\chi")

    return chi_mat
    
    
## Summary
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
  
### tests

def fct_test(x, **kwargs):
    return x