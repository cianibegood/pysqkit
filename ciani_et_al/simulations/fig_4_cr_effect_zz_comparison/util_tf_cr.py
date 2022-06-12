import numpy as np 
import cmath
import matplotlib.pyplot as plt 
from pysqkit import QubitSystem, Qubit
from pysqkit.util.linalg import get_mat_elem, tensor_prod, \
    hilbert_schmidt_prod
from typing import List, Callable, Dict
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
        ax.hlines(y=energy[level], xmin=x_min, xmax=x_max, 
                  linewidth=plot_setup['lw_levels'], color='k')
        ax.text(x_max+0.2, energy[level]-0.05, r'$ \vert' + level + 
                ' \\rangle$', fontsize=plot_setup['fs'], zorder=10)
        ax.tick_params(labelsize=plot_setup['ls'])
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    qubit_labels = system.labels
    n1n2_op = system[qubit_labels[0]].charge_op().dot(\
        system[qubit_labels[1]].charge_op())
    abs_mat_elem_vec = np.zeros(len(levels_to_plot)**2)
    count = 0
    for level_1 in levels_to_plot:
        for level_2 in levels_to_plot:
            abs_mat_elem_vec[count] = \
                np.abs(get_mat_elem(n1n2_op, state[level_1], state[level_2]))
            count += 1
    max_abs_mat_elem = np.max(abs_mat_elem_vec)
    for level_1 in levels_to_plot:
        for level_2 in levels_to_plot:
            abs_mat_elem = \
                np.abs(get_mat_elem(n1n2_op, state[level_1], state[level_2]))
            shade_factor = abs_mat_elem/max_abs_mat_elem
            if shade_factor > 1e-4 and np.abs(energy[level_1] - \
                energy[level_2]) < detuning_tol:
                ax.annotate('', (x_avg[level_1], energy[level_1]), \
                            (x_avg[level_2], energy[level_2]), 
                            arrowprops=dict(color='darkorange', 
                            alpha=shade_factor**5, 
                            lw=plot_setup['lw'], arrowstyle='<->'))
                
    if show_drive:
        ax.annotate('', (x_avg['10'], energy['10']), 
                    (x_avg['13'], energy['13']), 
                    arrowprops=dict(color='darkblue', lw=plot_setup['lw'],
                    arrowstyle='<->', linestyle = '--'))
        ax.annotate('', (x_avg['00'], energy['00']), 
                    (x_avg['03'], energy['03']), 
                    arrowprops=dict(color='darkblue', lw=plot_setup['lw'],
                    arrowstyle='<->', linestyle = '--'))
        
    ax.set_title(r'$\mathrm{Transmon} \leftarrow \quad \rightarrow'
                 r'\mathrm{Fluxonium}$', {'fontsize': plot_setup['fs']})
    ax.set_ylabel(r'$\mathrm{Energy} \, (\mathrm{GHz})$', 
                  fontsize=plot_setup['fs'])
    plt.show()  

def zz(system: QubitSystem) -> float:
    xi_zz = system.state('00')[0] + system.state('11')[0] \
        - system.state('01')[0] - system.state('10')[0]
    return xi_zz

def mu_yz_flx(
    comp_states: List[np.ndarray], 
    op: np.ndarray,
    eps: float
) -> float:
    """
    Description
    ---------------------------------------------------------------------------
    Evaluates the CR coefficient numerically in the dressed basis when
    driving the fluxonium
    """
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'])
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(yz0 - yz1))/2*eps/2

def mu_zy_transm(
    comp_states: List[np.ndarray], 
    op: np.ndarray,
    eps: float
) -> float:
    """
    Description
    ---------------------------------------------------------------------------
    Evaluates the CR coefficient numerically in the dressed basis when
    driving the transmon
    """
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['01'])
    yz1 = get_mat_elem(op, comp_states['10'], comp_states['11'] )
    return (np.imag(yz0 - yz1))/2

def mu_yi_flx(
    comp_states: List[np.ndarray], 
    op: np.ndarray,
    eps: float
) -> float:
    """
    Description
    ---------------------------------------------------------------------------
    Evaluates the direct drive on the transmon numerically in the dressed basis 
    when driving the fluxonium
    """
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'] )
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(yz0 + yz1))/2*eps/2

def mu_yz_flx_sw(
    transm: Qubit,
    flx: Qubit,
    jc: float,
    eps: float
):
    """
    Description
    ---------------------------------------------------------------------------
    Evaluates the CR coefficient via the second order Schrieffer-Wolff
    transformation
    """
    q_zpf = transm.charge_zpf
    omega_t = transm.freq
    omega_flx, states_flx = flx.eig_states(4)
    omega_flx = omega_flx - omega_flx[0]
    q_10 = np.imag(get_mat_elem(flx.charge_op(), states_flx[1], states_flx[0]))
    q_21 = np.imag(get_mat_elem(flx.charge_op(), states_flx[2], states_flx[1]))
    q_30 = np.imag(get_mat_elem(flx.charge_op(), states_flx[3], states_flx[0]))
    coeff = q_21**2/(omega_flx[2] - (omega_t + omega_flx[1]))
    coeff += -q_30**2/(omega_flx[3] - omega_t)
    coeff += q_10**2/(omega_t - omega_flx[1]) 
    mu_yz = jc*q_zpf*coeff/2*eps/2
    return mu_yz

def cr_gate_time(
    cr_coeff: float
):
    return 1/(2*np.pi*cr_coeff)*np.pi/4


def single_qubit_corrections(
    sup_op: np.ndarray,
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:
    # WARNING: AC I am really puzzled by this...if I use the analog
    # of this for single-qubit gate I get the wrong result and
    # we should actually use sigma_plus...???

    sigma_m1 = tensor_prod([np.array([[0.0, 0.0], [1.0, 0.0]]), 
                           np.array([[1.0, 0.0], [0.0, 0.0]])])
    sigma_m2 = tensor_prod([np.array([[1.0, 0.0], [0.0, 0.0]]), 
                           np.array([[0.0, 0.0], [1.0, 0.0]])])
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

