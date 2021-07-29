#-----------------------------------------------------------------------------
# File that contains auxiliary functions to simplify the calculations
# needed to study the two-qubit gate between a transmon and a fluxonium 
# using the |1_t 3_f>-|0_t 4_f> inspired by the fluxonium-fluxonium gate 
# described in Ficheux et al. Phys. Rev. X 11, 021026 (2021)
#-----------------------------------------------------------------------------

import numpy as np 
import cmath
import matplotlib.pyplot as plt 
from pysqkit import QubitSystem
from pysqkit.util.linalg import get_mat_elem, tensor_prod, \
    hilbert_schmidt_prod
from typing import List, Callable
import pysqkit.util.transformations as trf
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def energy_levels_diagram(
    system: QubitSystem,
    levels_to_plot: List[str],
    plot_setup={'fs': 20, 'lw': 2.0, 'lw_levels': 3.0, 'ls': 16},
    detuning_tol=3.0,
    show_drive=True
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ground_energy = system.state('00')[0]
    energy = {}
    state = {}
    step = 11.0
    level_width = 5.0
    x_avg = {}
    
    for level in levels_to_plot:
        energy[level] = system.state(level)[0] - ground_energy
        state[level] = system.state(level)[1]
    for level in levels_to_plot:
        level_as_list = [int(lev) for lev in level]
        x_min = -0.5 + step*level_as_list[1] - step*level_as_list[0]
        x_max = x_min + level_width
        x_avg[level] = (x_min + x_max)/2
        ax.hlines(y=energy[level], xmin=x_min, xmax=x_max, linewidth=plot_setup['lw_levels'], color='k')
        ax.text(x_max+0.2, energy[level]-0.05, r'$ \vert' + level + ' \\rangle$', 
                fontsize=plot_setup['fs'], zorder=10)
        ax.tick_params(labelsize=plot_setup['ls'])
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    qubit_labels = system.labels
    n1n2_op = system[qubit_labels[0]].charge_op().dot(system[qubit_labels[1]].charge_op())
    abs_mat_elem_vec = np.zeros(len(levels_to_plot)**2)
    count = 0
    for level_1 in levels_to_plot:
        for level_2 in levels_to_plot:
            abs_mat_elem_vec[count] = np.abs(get_mat_elem(n1n2_op, state[level_1], state[level_2]))
            count += 1
    max_abs_mat_elem = np.max(abs_mat_elem_vec)
    for level_1 in levels_to_plot:
        for level_2 in levels_to_plot:
            abs_mat_elem = np.abs(get_mat_elem(n1n2_op, state[level_1], state[level_2]))
            shade_factor = abs_mat_elem/max_abs_mat_elem
            if shade_factor > 1e-4 and np.abs(energy[level_1] - energy[level_2]) < detuning_tol:
                ax.annotate('', (x_avg[level_1], energy[level_1]), (x_avg[level_2], energy[level_2]), 
                            arrowprops=dict(color='darkorange', alpha=shade_factor**5,\
                                            lw=plot_setup['lw'], arrowstyle='<->'))
                
    if show_drive:
        ax.annotate('', (x_avg['10'], energy['10']), (x_avg['13'], energy['13']), 
                            arrowprops=dict(color='darkblue', lw=plot_setup['lw'],\
                                            arrowstyle='<->', linestyle = '--'))
        ax.annotate('', (x_avg['00'], energy['00']), (x_avg['03'], energy['03']), 
                            arrowprops=dict(color='darkblue', lw=plot_setup['lw'],\
                                            arrowstyle='<->', linestyle = '--'))
        
    ax.set_title(r'$\mathrm{Transmon} \leftarrow \quad \rightarrow  \mathrm{Fluxonium}$', \
                 {'fontsize': plot_setup['fs']})
    ax.set_ylabel(r'$\mathrm{Energy} \, (\mathrm{GHz})$', fontsize=plot_setup['fs'])
    plt.show()  

def zz(system: QubitSystem) -> float:
    xi_zz = system.state('00')[0] + system.state('11')[0] \
        - system.state('01')[0] - system.state('10')[0]
    return xi_zz

def delta(system: QubitSystem) -> float:
    delta_gate = (system.state('13')[0] - system.state('10')[0]) - \
        (system.state('03')[0] - system.state('00')[0])
    return delta_gate 

def single_qubit_corrections(
    sup_op: np.ndarray,
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:
    sigma_m1 = tensor_prod([np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 0.0]])])
    sigma_m2 = tensor_prod([np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])])
    sigma_m1_vec = trf.mat_to_vec(sigma_m1, hs_basis)
    sigma_m2_vec = trf.mat_to_vec(sigma_m2, hs_basis)
    evolved_sigma_m1_vec = sup_op.dot(sigma_m1_vec)
    evolved_sigma_m2_vec = sup_op.dot(sigma_m2_vec)
    evolved_sigma_m1 = trf.vec_to_mat(evolved_sigma_m1_vec, hs_basis)
    evolved_sigma_m2 = trf.vec_to_mat(evolved_sigma_m2_vec, hs_basis)
    phi10 = cmath.phase(hilbert_schmidt_prod(sigma_m1, evolved_sigma_m1))
    phi01 = cmath.phase(hilbert_schmidt_prod(sigma_m2, evolved_sigma_m2))
    p_phi10 = np.array([[1, 0], [0, np.exp(-1j*phi10)]])
    p_phi01 = np.array([[1, 0], [0, np.exp(-1j*phi01)]])
    return tensor_prod([p_phi10, p_phi01])