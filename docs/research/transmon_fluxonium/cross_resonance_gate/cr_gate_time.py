import numpy as np
import scipy.integrate
import time
import qutip as qtp
from scipy.optimize import minimize
import pysqkit
from pysqkit import QubitSystem
from pysqkit.drives.pulse_shapes import gaussian_top
from pysqkit.util.metrics import average_process_fidelity, \
    average_gate_fidelity
from pysqkit.util.phys import temperature_to_thermalenergy
import pysqkit.util.transformations as trf
from pysqkit.util.linalg import get_mat_elem
from pysqkit.util.hsbasis import weyl_by_index
import cmath
import multiprocessing
from functools import partial
import json
from typing import List, Dict, Callable
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import util_cr as util

def mu_yz_flx(comp_states, op) -> float:
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'])
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(yz0 - yz1))/2

def mu_yi_flx(comp_states, op) -> float:
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'] )
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(yz0 + yz1))/2

def func_to_minimize(
    pulse_time: list,
    t_rise: float,
    eps: float
) -> float:
    step = 1e-3
    n_points = int(pulse_time[0]/step)
    times = np.linspace(0, pulse_time[0], n_points)
    pulse = gaussian_top(times, t_rise, pulse_time[0])
    integral = scipy.integrate.simpson(2*np.pi*eps*pulse, times)
    return np.abs(integral - np.pi/2) 

def cry(theta):
    ide = np.identity(4)
    yz = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]]))
    return np.cos(theta/2)*ide - 1j*np.sin(theta/2)*yz

def ry(theta):
    rot_y = np.cos(theta/2)*np.identity(2) - \
        1j*np.sin(theta/2)*np.array([[0, -1j], [1j, 0]])
    return np.kron(rot_y, np.identity(2))

def get_fidelity_leakage(
    gate_time: float, 
    system: QubitSystem,
    t_rise: float,
    eps_drive:float,
    freq_drive: float,
    comp_states_list: List 
):
    pts_per_drive_period = 5.0

    nb_points = int(gate_time*freq_drive*pts_per_drive_period)
    tlist = np.linspace(0, gate_time, nb_points)

    system['F'].drives['cr_drive_f'].set_params(phase=0, 
                                                time=tlist, 
                                                rise_time=t_rise, 
                                                pulse_time=gate_time, 
                                                amp=eps_drive, 
                                                freq=freq_drive)
    
    simu_opt = qtp.solver.Options()
    simu_opt.atol = 1e-12
    simu_opt.rtol = 1e-10

    env_syst = pysqkit.tomography.TomoEnv(system=system, 
                                          time=2*np.pi*tlist, 
                                          options=simu_opt, 
                                          with_noise=False)
    env_syst_noise = pysqkit.tomography.TomoEnv(system=system, 
                                          time=2*np.pi*tlist, 
                                          options=simu_opt, 
                                          with_noise=True)
    
    #n_process = 1

    sup_op = env_syst.to_super(comp_states_list, weyl_by_index)
    sup_op_noise = env_syst_noise.to_super(comp_states_list, weyl_by_index)
    
    sq_corr = util.single_qubit_corrections(sup_op, weyl_by_index)
    sq_corr_sup = trf.kraus_to_super(sq_corr, weyl_by_index)
    total_sup_op = sq_corr_sup.dot(sup_op)

    sq_corr_noise = util.single_qubit_corrections(sup_op_noise, weyl_by_index)
    sq_corr_sup_noise = trf.kraus_to_super(sq_corr_noise, weyl_by_index)
    total_sup_op_noise = sq_corr_sup_noise.dot(sup_op_noise)
    
    cr_super_target = trf.kraus_to_super(cry(-np.pi/2), weyl_by_index)
    
    theta_list = list(np.linspace(0, 2*np.pi, 500))
    
    fid_list_ry = []
    fid_list_ry_noise = []
    for theta in theta_list:
        rot_y_super = trf.kraus_to_super(ry(theta), weyl_by_index)
        avg_tmp = average_process_fidelity(cr_super_target, 
                                           rot_y_super.dot(total_sup_op))
        avg_tmp_noise = average_process_fidelity(cr_super_target, 
                                                 rot_y_super.dot(total_sup_op_noise))
        fid_list_ry.append(avg_tmp)
        fid_list_ry_noise.append(avg_tmp_noise)
    
    fid_ry = np.array(fid_list_ry)
    fid_ry_noise = np.array(fid_list_ry_noise)
    
    max_index = np.argmax(fid_ry)
    max_index_noise = np.argmax(fid_ry_noise)
    sup_rot_y_opt = trf.kraus_to_super(ry(theta_list[max_index]), weyl_by_index)
    sup_rot_y_opt_noise = trf.kraus_to_super(ry(theta_list[max_index_noise]), weyl_by_index)
    avg_leakage = env_syst.leakage(comp_states_list)
    avg_leakage_noise = env_syst_noise.leakage(comp_states_list)
    
    f_gate = average_gate_fidelity(cr_super_target, 
                                   sup_rot_y_opt.dot(total_sup_op), avg_leakage)
    f_gate_noise = average_gate_fidelity(cr_super_target, 
                                        sup_rot_y_opt_noise.dot(total_sup_op_noise), 
                                        avg_leakage_noise)
    
    result = {"gate_time": gate_time, "gate_fid": f_gate, 
             "avg_leakage": avg_leakage, "gate_fid_noise": f_gate_noise,
             "avg_leakage_noise": avg_leakage_noise}
    
    return result    


def main():
    with open('../flx_transm_params.txt') as param_file:
        parameters_set = json.load(param_file)

        temperature = 0.020 # K
    thermal_energy = temperature_to_thermalenergy(temperature) # kb T/h in GHz

    p_set = "3"


    #Transmon
    levels_t = 3
    transm = pysqkit.qubits.SimpleTransmon(
        label='T', 
        max_freq=parameters_set[p_set]["max_freq_t"], 
        anharm=parameters_set[p_set]["anharm_t"],
        diel_loss_tan=parameters_set[p_set]["diel_loss_tan_t"], #set to zero to check d_1 L1 = d_2 L2
        env_thermal_energy=thermal_energy,    
        dim_hilbert=levels_t,
        dephasing_times=None #parameters_set[p_set]["dephasing_times_t"]
    )

    #Fluxonium
    levels_f = 5

    flx = pysqkit.qubits.Fluxonium(
        label='F', 
        charge_energy=parameters_set[p_set]["charge_energy_f"], 
        induct_energy=parameters_set[p_set]["induct_energy_f"], 
        joseph_energy=parameters_set[p_set]["joseph_energy_f"], #8.0, 
        diel_loss_tan=parameters_set[p_set]["diel_loss_tan_f"], #set to zero to check d_1 L1 = d_2 L2
        env_thermal_energy=thermal_energy,
        dephasing_times=None #parameters_set[p_set]["dephasing_times_f"] #ns/2*np.pi 
    )
    flx.diagonalize_basis(levels_f)

    # We also add a drive on the fluxonium
    flx.add_drive(
        pysqkit.drives.microwave_drive,
        label='cr_drive_f',
        pulse=pysqkit.drives.pulses.cos_modulation,
        pulse_shape=pysqkit.drives.pulse_shapes.gaussian_top
    )


    jc = parameters_set[p_set]["jc"]
    coupled_sys = transm.couple_to(flx, coupling=pysqkit.couplers.capacitive_coupling, strength=jc)

    state_label = ["00", "01", "10", "11"]
    comp_states = {}
    for label in state_label:
        state_tmp = coupled_sys.state(label)[1]
        loc = np.argmax(np.abs(state_tmp))
        phase = cmath.phase(state_tmp[loc])
        state_tmp = np.exp(-1j*phase)*state_tmp
        comp_states[label] = state_tmp

    eps_drive = 0.62 #GHz
    q_op = coupled_sys["F"].charge_op()
    freq_drive = transm.max_freq
    t_rise = 10.0 # [ns]

    t_tot_0 = [100.0]

    args_to_pass = (t_rise, np.abs(mu_yz_flx(comp_states, q_op))*eps_drive) 

    minimization_result = minimize(func_to_minimize, t_tot_0, args=args_to_pass)

    t_tot = minimization_result['x'][0]

    comp_states_list = []
    for key in comp_states.keys():
        comp_states_list.append(comp_states[key])
    
    n_points = 200
    gate_time_list = np.linspace(150,  170, n_points)

    func = partial(get_fidelity_leakage, system=coupled_sys, t_rise=t_rise, 
                   eps_drive=eps_drive, freq_drive=freq_drive, 
                   comp_states_list=comp_states_list)

    n_process = 200

    start = time.time()
        
    pool = multiprocessing.Pool(processes=n_process)
        
    result = pool.map(func, gate_time_list, chunksize=int(n_points//n_process))
        
    pool.close()
    pool.join()  

    end = time.time()

    print("Computation time = {} s".format(end - start))

    save = True
    if save:
        with open("tmp/fid_leak_gate_time_p_set_" + p_set + ".txt", "w") as fp:
            json.dump(result, fp)
        paramfile = open('tmp/cr_gate_time_params_p_set_' + str(p_set) + '.txt', "w+")
        paramfile.write('The data where generated using ')
        paramfile.write('the following drive parameters. \n ')
        paramfile.write('t_rise = %f ns' % t_rise + '\n')
        paramfile.write('freq_drive = %f GHz' % freq_drive + '\n')
        paramfile.write('eps_drive = %f GHz' % eps_drive + '\n')
        paramfile.write('parameter_set:\n')
        paramfile.write(str(parameters_set[p_set]))


if __name__ == '__main__':
    main()
    


