import numpy as np
import scipy.integrate
import time
import qutip as qtp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pysqkit
from pysqkit import QubitSystem
from pysqkit.drives.pulse_shapes import gaussian_top
from pysqkit.util.metrics import average_process_fidelity, \
    average_gate_fidelity
from pysqkit.util.phys import temperature_to_thermalenergy
from pysqkit.util.quantum import generalized_rabi_frequency
import pysqkit.util.transformations as trf
from pysqkit.util.linalg import get_mat_elem
from pysqkit.solvers.solvkit import integrate
from pysqkit.util.hsbasis import weyl_by_index
from pysqkit.solvers import solvkit
from pysqkit.drives.pulse_shapes import gaussian_top
import qutip
from typing import List, Dict, Callable
import multiprocessing
import util_cr as util
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import copy
import json
import cmath

from IPython.display import display, Latex

def mu_yz_flx(comp_states, op) -> float:
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'])
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'])
    return (np.imag(yz0 - yz1))/2

def mu_zy_transm(comp_states, op) -> float:
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['01'])
    yz1 = get_mat_elem(op, comp_states['10'], comp_states['11'] )
    return (np.imag(yz0 - yz1))/2

def mu_yi_flx(comp_states, op) -> float:
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'] )
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(yz0 + yz1))/2

def func_to_minimize(
    eps: float,
    pulse_time: float,
    t_rise: float,
    cr_coeff: float
) -> float:
    step = 1e-3
    n_points = int(pulse_time/step)
    times = np.linspace(0, pulse_time, n_points)
    pulse = gaussian_top(times, t_rise, pulse_time)
    integral = scipy.integrate.simpson(2*np.pi*(eps*cr_coeff/2)*pulse, times)
    return np.abs(integral - np.pi/4)  #Watch out factor of 2?  

def cry(theta):
    ide = np.identity(4)
    yz = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]]))
    return np.cos(theta/2)*ide - 1j*np.sin(theta/2)*yz

def crx(theta):
    ide = np.identity(4)
    zx = np.kron(np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]]))
    return np.cos(theta/2)*ide - 1j*np.sin(theta/2)*zx

def ry_t(theta):
    rot_y = np.cos(theta/2)*np.identity(2) - \
        1j*np.sin(theta/2)*np.array([[0, -1j], [1j, 0]])
    return np.kron(rot_y, np.identity(2))

def ry_f(theta):
    rot_y = np.cos(theta/2)*np.identity(2) - \
        1j*np.sin(theta/2)*np.array([[0, -1j], [1j, 0]])
    return np.kron(np.identity(2), rot_y)

def optimal_sup_op(
    sup_op_target: np.ndarray,
    sup_op: np.ndarray    
):
    sq_corr = util.single_qubit_corrections(sup_op, weyl_by_index)
    sq_corr_sup = trf.kraus_to_super(sq_corr, weyl_by_index)
    total_sup_op = sq_corr_sup.dot(sup_op)
    fid_list_ry = []
    theta_list = list(np.linspace(0, 2*np.pi, 100))
    for theta in theta_list:
        rot_y_super = trf.kraus_to_super(ry_t(theta), weyl_by_index)
        fid_list_ry.append(average_process_fidelity(sup_op_target, rot_y_super.dot(total_sup_op)))

    fid_ry = np.array(fid_list_ry)

    max_index = np.argmax(fid_ry)
    sup_rot_y_opt = trf.kraus_to_super(ry_t(theta_list[max_index]), weyl_by_index)
    total_sup_op_ry = sup_rot_y_opt.dot(total_sup_op)
    
    return total_sup_op_ry

    def get_fidelity_leakage(
    transm_freq: float
) -> dict:
    with open('../flx_transm_params.txt') as param_file:
        parameters_set = json.load(param_file)
    temperature = 0.020 #0.020 # K
    thermal_energy = temperature_to_thermalenergy(temperature) # kb T/h in GHz
    d_comp = 4

    p_set = "2"


    #Transmon
    levels_t = 3
    transm = pysqkit.qubits.SimpleTransmon(
        label='T', 
        max_freq=transm_freq, #parameters_set[p_set]["max_freq_t"], 
        anharm=parameters_set[p_set]["anharm_t"],
        diel_loss_tan=parameters_set[p_set]["diel_loss_tan_t"],
        env_thermal_energy=thermal_energy,    
        dim_hilbert=levels_t,
        dephasing_times=None 
    )

    #Fluxonium
    levels_f = 4

    flx = pysqkit.qubits.Fluxonium(
        label='F', 
        charge_energy=parameters_set[p_set]["charge_energy_f"], 
        induct_energy=parameters_set[p_set]["induct_energy_f"], 
        joseph_energy=parameters_set[p_set]["joseph_energy_f"], 
        diel_loss_tan=parameters_set[p_set]["diel_loss_tan_f"], 
        env_thermal_energy=thermal_energy,
        dephasing_times=None 
    )
    flx.diagonalize_basis(levels_f)

    # We also add a drive on the fluxonium
    flx.add_drive(
        pysqkit.drives.microwave_drive,
        label='cr_drive_f',
        pulse=pysqkit.drives.pulses.cos_modulation,
        pulse_shape=pysqkit.drives.pulse_shapes.gaussian_top
    )

    d_leak = levels_t*levels_f - d_comp

    jc = parameters_set[p_set]["jc"]
    coupled_sys = transm.couple_to(flx, coupling=pysqkit.couplers.capacitive_coupling, strength=jc)
    

    states_label = coupled_sys.all_state_labels()
    states_dict = coupled_sys.states_as_dict(as_qobj=True)
    
    state_label = ["00", "01", "10", "11"]
    comp_states = {}
    for label in state_label:
        state_tmp = coupled_sys.state(label)[1]
        loc = np.argmax(np.abs(state_tmp))
        phase = cmath.phase(state_tmp[loc])
        state_tmp = np.exp(-1j*phase)*state_tmp
        comp_states[label] = state_tmp
    
    q_op = coupled_sys["F"].charge_op()
    freq_drive = transm.max_freq
    t_rise = 10.0 # [ns]
    t_gate = 130.0
    cr_coeff = np.abs(mu_yz_flx(comp_states, q_op))

    eps_0 = 0.6

    args_to_pass = (t_gate, t_rise, cr_coeff) #factor of two seems right here

    # We find the total time to obtain the desired gate

    start = time.time()

    minimization_result = minimize(func_to_minimize, eps_0, args=args_to_pass)

    print(minimization_result)

    end = time.time()

    eps_drive = minimization_result['x'][0] #1/(util.y_z_flx(coupled_sys, 'F')*eps_drive*4)  # [ns]
    pts_per_drive_period = 10

    #t_tot = 135

    nb_points = int(t_gate*freq_drive*pts_per_drive_period)
    tlist = np.linspace(0, t_gate, nb_points)

    coupled_sys['F'].drives['cr_drive_f'].set_params(phase=0, time=tlist, rise_time=t_rise, pulse_time=t_gate,
                                                     amp=eps_drive, freq=freq_drive)
    
    simu_opt = qtp.solver.Options()
    simu_opt.atol = 1e-12
    simu_opt.rtol = 1e-10

    env_syst = pysqkit.tomography.TomoEnv(system=coupled_sys, time=2*np.pi*tlist, 
                                          options=simu_opt, with_noise=False)
    
    env_syst_noisy = pysqkit.tomography.TomoEnv(system=coupled_sys, time=2*np.pi*tlist, 
                                          options=simu_opt, with_noise=True)
    
    comp_states_list = []
    for key in comp_states.keys():
        comp_states_list.append(comp_states[key])
    
    avg_leakage = env_syst.leakage(comp_states_list)
    avg_leakage_noisy = env_syst_noisy.leakage(comp_states_list)
    
    sup_op = env_syst.to_super(comp_states_list, weyl_by_index)
    sup_op_noisy = env_syst_noisy.to_super(comp_states_list, weyl_by_index)
    
    cr_super_target = trf.kraus_to_super(cry(-np.pi/2), weyl_by_index)
    
    opt_sup_op = optimal_sup_op(cr_super_target, sup_op)
    opt_sup_op_noisy = optimal_sup_op(cr_super_target, sup_op_noisy)
    
    f_gate = average_gate_fidelity(cr_super_target, opt_sup_op, avg_leakage)
    f_gate_noisy = average_gate_fidelity(cr_super_target, opt_sup_op_noisy, avg_leakage_noisy)
    
    res={}
    res["transm_freq"] = transm_freq 
    res["L1"] = avg_leakage
    res["L1_noisy"] = avg_leakage_noisy
    res["fidelity"] = f_gate
    res["fidelity_noisy"] = f_gate_noisy
    
    return res

def main():
    n_points = 2
    n_processes = 2
    freq_list = np.linspace(4.2, 5.8, n_points)

    start = time.time()

    pool = multiprocessing.Pool(processes=n_processes)

    result = pool.map(get_fidelity_leakage, freq_list, chunksize=int(n_points//n_processes))

    pool.close()
    pool.join()

    end=time.time()

    print("Computation time: {} s".format(end - start))

    save = True
    if save:
        with open("/tmp/cr_fidelity_leakage.txt", "w") as fp:
            json.dump(result, fp)

if __name__ == "__main__":
    main()