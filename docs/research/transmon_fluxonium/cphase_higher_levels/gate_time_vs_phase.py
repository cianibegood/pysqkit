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
    cond_phase: float,
    system: QubitSystem
):
    x0 = np.array([0.017, 7.15]) #initial guess
    qubit_labels = system.labels
    eps_ratios = {qubit_labels[0]: 0.0, qubit_labels[1]:1.0}
    args_to_pass = (['00', '03'], ['10', '13'], system, 
                    cond_phase, eps_ratios) 

    minimization_result = minimize(func_to_minimize, x0, args=args_to_pass)

    eps_drive = minimization_result['x'][0]
    freq_drive = minimization_result['x'][1]

    res = {}
    res["cond_phase"] = cond_phase 
    res["eps_drive"] = eps_drive
    res["freq_drive"] = freq_drive

    eps = {}
    for qubit in qubit_labels:
        eps[qubit] = eps_drive*eps_ratios[qubit]
    rabi_period = 1/generalized_rabi_frequency(["00", "03"], 
                                               eps, freq_drive, system)

    t_rise = 5.0 # [ns]

    t_gate_0 = [rabi_period]

    args_to_pass = (t_rise, rabi_period) 

    minimization_result_time = minimize(func_to_minimize_time, 
                                        t_gate_0, args=args_to_pass)

    t_gate = minimization_result_time['x'][0]

    res["t_gate"] = t_gate

    return res



def main():
    with open('flx_transm_params.txt') as param_file:
        parameters_set = json.load(param_file)
    
    temperature = 0.020 # K
    thermal_energy = temperature_to_thermalenergy(temperature) # kb T/h in GHz
    d_comp = 4

    p_set = "0"


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
    
    phase_in = np.pi
    phase_fin = 3*np.pi
    n_points = 4
    cond_phase_list = list(np.linspace(phase_in, phase_fin, n_points))

    n_process = 4

    print(n_process)
    print(n_points)

    func = partial(get_result, system=coupled_sys)

    start = time.time()

    pool = multiprocessing.Pool(processes=n_process)

    result = pool.map(func, cond_phase_list, 
                      chunksize=int(n_points//n_process))

    pool.close()
    pool.join()

    end = time.time()

    print("Computation time: {} s".format(end - start))

    save = False
    if save:
        with open('tmp/gate_time_phase_result.txt', 'w') as my_file:
            json.dump(result, my_file)
        with open('tmp/gate_time_phase_params.txt', 'w') as my_file:
            json.dump(parameters_set["0"], my_file)

if __name__ == '__main__':
    main()