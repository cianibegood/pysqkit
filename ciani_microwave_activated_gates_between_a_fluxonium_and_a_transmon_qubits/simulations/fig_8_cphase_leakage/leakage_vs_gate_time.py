import numpy as np
import time
import qutip as qtp
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.optimize import minimize
import pysqkit
from pysqkit import QubitSystem
from pysqkit.util.metrics import average_process_fidelity, \
    average_gate_fidelity
from pysqkit.drives.pulse_shapes import gaussian_top
from pysqkit.util.phys import temperature_to_thermalenergy
from pysqkit.util.quantum import generalized_rabi_frequency
import pysqkit.util.transformations as trf
from pysqkit.util.hsbasis import weyl_by_index
from typing import List, Dict, Callable
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import util_cphase as util
import cmath
import json
import multiprocessing
from functools import partial


def func_to_minimize(
    x0: np.ndarray,
    levels_first_transition: List['str'],
    levels_second_transition: List['str'],
    system: QubitSystem,
    cond_phase: float,
    eps_ratio_dict: Dict    
) -> float:
    
    """
    Description
    --------------------------------------------------------------------------
    Function to minimize in order to match the parameters to 
    implement a CPHASE gate given a certain conditional phase up to 
    single-qubit rotations. 

    Parameters
    --------------------------------------------------------------------------
    x0 : np.ndarray([eps_reference, drive_freq]) 
        It represents the parameters to be minimized.
    levels_first_transition : List['str'] 
        List with the labels of the first transition whose generalized Rabi 
        frequency has to be matched
    levels_second_transition : List['str'] 
        List with the labels of the second transition whose generalized Rabi 
        frequency has to be matched
    system: QubitSystem
        The coupled system we are analyzing
    cond_phase: float
        Conditional phase
    eps_ratio_dict: Dict 
        Dictionary whose keys are system.labels. The entries correspond
        to the ratios between the corresponding qubit drive and the 
        reference drive.     
    """
    
    qubit_labels = system.labels
    eps = {}
    for qubit in qubit_labels:
        eps[qubit] = x0[0]*eps_ratio_dict[qubit]
    rabi_first_transition = generalized_rabi_frequency(levels_first_transition, 
                                                       eps, x0[1], system)
    rabi_second_transition = generalized_rabi_frequency(levels_second_transition, 
                                                        eps, x0[1], system)
    delta_gate = util.delta(system)
    y = np.sqrt( (rabi_first_transition - rabi_second_transition)**2 + \
                rabi_first_transition**2*(cond_phase - \
                    delta_gate/rabi_first_transition*np.pi)**2)
    return np.abs(y/delta_gate)

def func_to_minimize_time(
    pulse_time: list,
    t_rise: float,
    rabi_period
) -> float:
    """
    Description
    --------------------------------------------------------------------------
    Function to minimize to match the gate time with a gaussian pulse
    to the rabi period.
    """
    step = 1e-3
    n_points = int(pulse_time[0]/step)
    times = np.linspace(0, pulse_time[0], n_points)
    pulse = gaussian_top(times, t_rise, pulse_time[0])
    integral = scipy.integrate.simpson(pulse, times)
    return np.abs(integral - rabi_period)

def get_result(
    gate_time: List,
    system: QubitSystem,
    eps_drive: float,
    freq_drive: float, 
    rise_time: float,
    comp_states_list: List
):
    pts_per_drive_period = 10
    
    nb_points = int(gate_time*freq_drive*pts_per_drive_period)
    t_list = np.linspace(0, gate_time, nb_points)
    
    system['F'].drives['cz_drive_f'].set_params(phase=0, time=t_list, 
                                                     rise_time=rise_time, 
                                                     pulse_time=gate_time,
                                                     amp=eps_drive, 
                                                     freq=freq_drive)
    
    simu_opt = qtp.solver.Options()
    simu_opt.atol = 1e-12
    simu_opt.rtol = 1e-10
    env_syst = pysqkit.tomography.TomoEnv(system=system, 
                                          time=2*np.pi*t_list, 
                                          options=simu_opt, 
                                          with_noise=False)
    env_syst_noise = pysqkit.tomography.TomoEnv(system=system, 
                                                time=2*np.pi*t_list, 
                                                options=simu_opt, 
                                                with_noise=True)
    
    avg_leakage = env_syst.leakage(comp_states_list)
    avg_leakage_noise = env_syst_noise.leakage(comp_states_list)
    
    return avg_leakage, avg_leakage_noise

def get_fidelity(
    system: QubitSystem,
    cond_phase:float,
    gate_time: float,
    avg_leakage: float,
    eps_drive: float,
    freq_drive: float,
    rise_time: float,
    comp_states_list: List,
    with_noise: bool  
):
    pts_per_drive_period = 10
    nb_points = int(gate_time*freq_drive*pts_per_drive_period)
    tlist = np.linspace(0, gate_time, nb_points)

    system['F'].drives['cz_drive_f'].set_params(phase=0, time=tlist, 
                                                rise_time=rise_time, 
                                                pulse_time=gate_time,
                                                amp=eps_drive, freq=freq_drive)
    
    simu_opt = qtp.solver.Options()
    simu_opt.atol = 1e-12
    simu_opt.rtol = 1e-10
    env_syst = pysqkit.tomography.TomoEnv(system=system, 
                                            time=2*np.pi*tlist, 
                                            options=simu_opt, 
                                            with_noise=with_noise)
    
    n_process = 4

    sup_op = env_syst.to_super(comp_states_list, weyl_by_index, n_process)

    sq_corr = util.single_qubit_corrections(sup_op, weyl_by_index)
    sq_corr_sup = trf.kraus_to_super(sq_corr, weyl_by_index)
    total_sup_op = sq_corr_sup.dot(sup_op)

    cphase = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], \
        [0, 0, 0, np.exp(1j*cond_phase)]])
    cphase_super = trf.kraus_to_super(cphase, weyl_by_index)

    f_gate = average_gate_fidelity(cphase_super, total_sup_op, 
                                   avg_leakage)
    
    return f_gate
    

def main():
    with open('../../flx_transm_params.txt') as param_file:
        parameters_set = json.load(param_file)
    
    temperature = 0.020 # K
    thermal_energy = temperature_to_thermalenergy(temperature) # kb T/h in GHz
    d_comp = 4

    p_set = "CPHASE"


    #Transmon
    levels_t = 3
    transm = pysqkit.qubits.SimpleTransmon(
        label='T', 
        max_freq=parameters_set[p_set]["max_freq_t"], 
        anharm=parameters_set[p_set]["anharm_t"],
        diel_loss_tan=parameters_set[p_set]["diel_loss_tan_t"],
        env_thermal_energy=thermal_energy,    
        dim_hilbert=levels_t,
        dephasing_times=parameters_set[p_set]["dephasing_times_t"]
    )

    #Fluxonium
    levels_f = 5

    flx = pysqkit.qubits.Fluxonium(
        label='F', 
        charge_energy=parameters_set[p_set]["charge_energy_f"], 
        induct_energy=parameters_set[p_set]["induct_energy_f"], 
        joseph_energy=parameters_set[p_set]["joseph_energy_f"],  
        diel_loss_tan=parameters_set[p_set]["diel_loss_tan_f"], 
        env_thermal_energy=thermal_energy,
        dephasing_times= parameters_set[p_set]["dephasing_times_f"]  
    )
    flx.diagonalize_basis(levels_f)

    # We also add a drive on the fluxonium
    flx.add_drive(
        pysqkit.drives.microwave_drive,
        label='cz_drive_f',
        pulse=pysqkit.drives.pulses.cos_modulation,
        pulse_shape=pysqkit.drives.pulse_shapes.gaussian_top
    )

    d_leak = levels_t*levels_f - d_comp

    jc = parameters_set[p_set]["jc"]
    coupled_sys = transm.couple_to(flx, 
                                   coupling=pysqkit.couplers.capacitive_coupling, 
                                   strength=jc)
    
    state_label = ["00", "01", "10", "11"]
    comp_states = {}
    for label in state_label:
        state_tmp = coupled_sys.state(label)[1]
        loc = np.argmax(np.abs(state_tmp))
        phase = cmath.phase(state_tmp[loc])
        state_tmp = np.exp(-1j*phase)*state_tmp
        comp_states[label] = state_tmp
    comp_states_list = []
    for key in comp_states.keys():
        comp_states_list.append(comp_states[key])
    
    cond_phase_list = [np.pi, 5/4*np.pi, 3/2*np.pi, 7/4*np.pi]

    output = []

    qubit_labels = coupled_sys.labels

    x0 = np.array([0.017, 7.15]) #initial guess 
    eps_ratios = {qubit_labels[0]: 0.0, qubit_labels[1]:1.0}

    for cond_phase in cond_phase_list:
        
        res_dict = {}
        
        res_dict["cond_phase"] = cond_phase
    
        args_to_pass = (['00', '03'], ['10', '13'], coupled_sys, 
                        cond_phase, eps_ratios) 

        minimization_result = minimize(func_to_minimize, x0, args=args_to_pass)

        eps_drive = minimization_result['x'][0]
        freq_drive = minimization_result['x'][1]
        
        eps = {}
        for qubit in qubit_labels:
            eps[qubit] = eps_drive*eps_ratios[qubit]
        rabi_period = 1/generalized_rabi_frequency(["00", "03"], 
                                                eps, freq_drive, coupled_sys)

        t_rise = 10.0 # [ns]
        
        res_dict["t_rise"] = t_rise

        t_gate_0 = [rabi_period]

        args_to_pass = (t_rise, rabi_period) 

        minimization_result_time = minimize(func_to_minimize_time, 
                                            t_gate_0, args=args_to_pass)

        t_gate_ideal = minimization_result_time['x'][0]
        
        res_dict["t_gate_ideal"] = t_gate_ideal
        
        delta_time = 5 #ns
        n_points = 50
        gate_time_list = list(np.linspace(t_gate_ideal - delta_time, 
                                          t_gate_ideal + delta_time, n_points))
        
        res_dict["gate_time_list"] = gate_time_list
        
        func = partial(get_result, system=coupled_sys, eps_drive=eps_drive,
                       freq_drive=freq_drive, rise_time=t_rise, 
                       comp_states_list=comp_states_list)

        n_process = 8

        start = time.time()
        
        pool = multiprocessing.Pool(processes=n_process)
        
        result = pool.map(func, gate_time_list, 
                          chunksize=int(n_points//n_process))

        pool.close()
        pool.join()

        end = time.time()

        print(result)

        print("Computation time = {} s".format(end - start) )
        
        avg_leak = []
        avg_leak_noise = []
        for res in result:
            avg_leak.append(res[0])
            avg_leak_noise.append(res[1])
        
        res_dict["avg_leakage"] = avg_leak
        res_dict["avg_leakage_noise"] = avg_leak_noise

        argmin_leak = np.argmin(avg_leak)
        argmin_leak_noise = np.argmin(avg_leak_noise)

        gate_time = gate_time_list[argmin_leak]
        gate_time_noise = gate_time_list[argmin_leak_noise]

        f_gate = get_fidelity(coupled_sys, cond_phase, gate_time,
                              avg_leak[argmin_leak], 
                              eps_drive, freq_drive, 
                              t_rise, comp_states_list, 
                              with_noise=False)
        
        f_gate_noise = get_fidelity(coupled_sys, cond_phase, 
                                    gate_time_noise,
                                    avg_leak_noise[argmin_leak_noise], 
                                    eps_drive, freq_drive, 
                                    t_rise, comp_states_list, 
                                    with_noise=True)
        
        res_dict["gate_fid"] = f_gate
        res_dict["gate_fid_noise"] = f_gate_noise
        output.append(res_dict)
        
        
    
    save = True
    if save:
        with open ('leakage_vs_gate_time_data.txt', "w") as my_file:
            json.dump(output, my_file)
        with open('leakage_vs_gate_time_params.txt', 'w') as my_file:
            json.dump(parameters_set["0"], my_file)
    
if __name__ == '__main__':
    main()