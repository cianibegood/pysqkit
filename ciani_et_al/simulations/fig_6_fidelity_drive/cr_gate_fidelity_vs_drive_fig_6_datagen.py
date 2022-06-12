import numpy as np
import scipy.integrate
import time
import qutip as qtp
from scipy.optimize import minimize
import pysqkit
from pysqkit.drives.pulse_shapes import gaussian_top
from pysqkit.util.metrics import average_process_fidelity, \
    average_gate_fidelity
from pysqkit.util.phys import temperature_to_thermalenergy
import pysqkit.util.transformations as trf
from pysqkit.util.hsbasis import pauli_by_index
import json
import cmath
import util_tf_cr

def func_to_minimize(
    pulse_time: list,
    t_rise: float,
    cr_coeff: float
) -> float:
    step = 1e-3
    n_points = int(pulse_time[0]/step)
    times = np.linspace(0, pulse_time[0], n_points)
    pulse = gaussian_top(times, t_rise, pulse_time[0])
    integral = scipy.integrate.simpson(2*np.pi*cr_coeff*pulse, times)
    return np.abs(integral - np.pi/4) 

def ry_t(theta):
    rot_y = np.cos(theta/2)*np.identity(2) - \
        1j*np.sin(theta/2)*np.array([[0, -1j], [1j, 0]])
    return np.kron(rot_y, np.identity(2))

def rx_t(theta):
    rot_x = np.cos(theta/2)*np.identity(2) - \
        1j*np.sin(theta/2)*np.array([[0, 1], [1, 0]])
    return np.kron(rot_x, np.identity(2))

def ry_f(theta):
    rot_y = np.cos(theta/2)*np.identity(2) - \
        1j*np.sin(theta/2)*np.array([[0, -1j], [1j, 0]])
    return np.kron(np.identity(2), rot_y)

def cry(theta):
    ide = np.identity(4)
    yz = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]]))
    return np.cos(theta/2)*ide - 1j*np.sin(theta/2)*yz

def crx(theta):
    ide = np.identity(4)
    zx = np.kron(np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]]))
    return np.cos(theta/2)*ide - 1j*np.sin(theta/2)*zx

def get_fidelity(
    eps: float,
    p_set: str
):
    with open('../../flx_transm_params.txt') as param_file:
        parameters_set = json.load(param_file)
    
    temperature = 0.020 #0.020 # K
    thermal_energy = temperature_to_thermalenergy(temperature) 
    d_comp = 4

    levels_t = 3
    transm = pysqkit.qubits.SimpleTransmon(
        label='T', 
        max_freq=parameters_set[p_set]["max_freq_t"], 
        anharm=parameters_set[p_set]["anharm_t"],
        diel_loss_tan=parameters_set[p_set]["diel_loss_tan_t"],
        env_thermal_energy=thermal_energy,    
        dim_hilbert=levels_t,
        dephasing_times=None 
    )

    levels_f = 6 # 6 for data in the paper
    
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

    flx.add_drive(
        pysqkit.drives.microwave_drive,
        label='cr_drive_f',
        pulse=pysqkit.drives.pulses.cos_modulation,
        pulse_shape=pysqkit.drives.pulse_shapes.gaussian_top
    )

    jc = parameters_set[p_set]["jc"]
    coupled_sys = \
        transm.couple_to(flx, 
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
        
    op = coupled_sys["F"].charge_op()
    freq_drive = transm.max_freq
    cr_coeff = np.abs(util_tf_cr.mu_yz_flx(comp_states, op, eps))
    t_rise = 10.0 # [ns]

    t_gate_0 = [util_tf_cr.cr_gate_time(cr_coeff)]

    args_to_pass = (t_rise, cr_coeff)

    start = time.time()

    minimization_result = minimize(func_to_minimize, t_gate_0, 
                                   args=args_to_pass)

    end = time.time()

    t_gate = minimization_result['x'][0] 
    pts_per_drive_period = 10

    nb_points = int(t_gate*freq_drive*pts_per_drive_period)
    
    nb_points_echo = int(t_gate/2*freq_drive*pts_per_drive_period)
    
    tlist = np.linspace(0, t_gate, nb_points)

    coupled_sys['F'].drives['cr_drive_f'].set_params(phase=0, time=tlist, 
                                                     rise_time=t_rise, 
                                                     pulse_time=t_gate, 
                                                     amp=eps, freq=freq_drive)
    
    simu_opt = qtp.solver.Options()
    simu_opt.atol = 1e-14
    simu_opt.rtol = 1e-12
    
    res = {}
    
    res["gate_time"] = t_gate
    res["transm_freq"] = transm.freq
    res["eps"] = eps
    
    my_hs_basis = pauli_by_index

    n_process = 1 # set to desired number of processes (we used 16 in the paper)
    
    cr_super_target = trf.kraus_to_super(cry(-np.pi/2), my_hs_basis)
    
    for noise in [True, False]:
        env_syst = pysqkit.tomography.TomoEnv(system=coupled_sys, 
                                              time=2*np.pi*tlist, 
                                              options=simu_opt, 
                                              with_noise=noise, 
                                              dressed_noise=False)

        sup_op = env_syst.to_super(comp_states_list, my_hs_basis, 
                                   n_process, speed_up=True)
        
        sq_corr = util_tf_cr.single_qubit_corrections(sup_op, my_hs_basis)
        sq_corr_sup = trf.kraus_to_super(sq_corr, my_hs_basis)
        total_sup_op = sq_corr_sup.dot(sup_op)
        
        theta_list = list(np.linspace(0, 2*np.pi, 100))
        
        fid_list_ry = []
        for theta in theta_list:
            rot_y_super = trf.kraus_to_super(ry_t(theta), my_hs_basis)
            fid_list_ry.append(\
                average_process_fidelity(cr_super_target, 
                                         rot_y_super.dot(total_sup_op)))

        fid_ry = np.array(fid_list_ry)

        max_fid = np.max(fid_ry)
        max_index = np.argmax(fid_ry)
        sup_rot_y_opt = trf.kraus_to_super(ry_t(theta_list[max_index]), 
                                                my_hs_basis)
        
        avg_leakage = env_syst.leakage(comp_states_list)
        
        total_sup_op_ry = sup_rot_y_opt.dot(total_sup_op)
        
        if noise:
            res["F_pro_noisy"] = max_fid
            res["L1_noisy"] = avg_leakage
            res["F_gate_noisy"] = \
                average_gate_fidelity(cr_super_target, 
                                      total_sup_op_ry, avg_leakage)
        else:
            res["F_pro"] = max_fid
            res["L1"] = avg_leakage
            res["F_gate"] = \
                average_gate_fidelity(cr_super_target, 
                                      total_sup_op_ry, avg_leakage)
    
    for noise in [True, False]:
        
        flx_a = pysqkit.qubits.Fluxonium(
            label='F_A', 
            charge_energy=parameters_set[p_set]["charge_energy_f"], 
            induct_energy=parameters_set[p_set]["induct_energy_f"], 
            joseph_energy=parameters_set[p_set]["joseph_energy_f"],  
            diel_loss_tan=parameters_set[p_set]["diel_loss_tan_f"], 
            env_thermal_energy=thermal_energy,
            dephasing_times=None 
        )
        flx_a.diagonalize_basis(levels_f)

        flx_a.add_drive(
            pysqkit.drives.microwave_drive,
            label='cr_drive_f_1',
            pulse=pysqkit.drives.pulses.cos_modulation,
            pulse_shape=pysqkit.drives.pulse_shapes.gaussian_top
        )

        
        tlist_echo = np.linspace(0, t_gate/2, nb_points_echo)
        
        coupled_sys_a = \
            transm.couple_to(flx_a, 
                             coupling=pysqkit.couplers.capacitive_coupling, 
                             strength=jc)

        coupled_sys_a['F_A'].drives['cr_drive_f_1'].set_params(phase=0, 
                                                               time=tlist_echo, 
                                                               rise_time=t_rise, 
                                                               pulse_time=t_gate/2, 
                                                               amp=eps, 
                                                               freq=freq_drive)

        env_syst_a = pysqkit.tomography.TomoEnv(system=coupled_sys_a, 
                                                time=2*np.pi*tlist_echo, 
                                                options=simu_opt, 
                                                with_noise=noise, 
                                                dressed_noise=False)

        sup_op_first = env_syst_a.to_super(comp_states_list, 
                                           my_hs_basis, n_process, 
                                           speed_up=True)

        sq_corr_first = util_tf_cr.single_qubit_corrections(sup_op_first, 
                                                            my_hs_basis)
        sq_corr_sup_first = trf.kraus_to_super(sq_corr_first, 
                                               my_hs_basis)
        sup_op_first_tot = sq_corr_sup_first.dot(sup_op_first)


        flx_b = pysqkit.qubits.Fluxonium(
            label='F_B', 
            charge_energy=parameters_set[p_set]["charge_energy_f"], 
            induct_energy=parameters_set[p_set]["induct_energy_f"], 
            joseph_energy=parameters_set[p_set]["joseph_energy_f"],  
            diel_loss_tan=parameters_set[p_set]["diel_loss_tan_f"], 
            env_thermal_energy=thermal_energy,
            dephasing_times=None 
            )
        flx_b.diagonalize_basis(levels_f)

        flx_b.add_drive(
            pysqkit.drives.microwave_drive,
            label='cr_drive_f_2',
            pulse=pysqkit.drives.pulses.cos_modulation,
            pulse_shape=pysqkit.drives.pulse_shapes.gaussian_top
        )

        coupled_sys_b = \
            transm.couple_to(flx_b, 
                             coupling=pysqkit.couplers.capacitive_coupling, 
                             strength=jc)

        coupled_sys_b['F_B'].drives['cr_drive_f_2'].set_params(phase=0, 
                                                               time=tlist_echo, 
                                                               rise_time=t_rise, 
                                                               pulse_time=t_gate/2, 
                                                               amp=-eps, freq=freq_drive)

        env_syst_b = pysqkit.tomography.TomoEnv(system=coupled_sys_b, 
                                                time=2*np.pi*tlist_echo, 
                                                options=simu_opt, 
                                                with_noise=noise, 
                                                dressed_noise=False)

        sup_op_second = env_syst_b.to_super(comp_states_list, 
                                            my_hs_basis, 
                                            n_process, 
                                            speed_up=True)

        sq_corr_second = util_tf_cr.single_qubit_corrections(sup_op_second, 
                                                             my_hs_basis)
        sq_corr_sup_second = trf.kraus_to_super(sq_corr_second, my_hs_basis)
        sup_op_second_tot = sq_corr_sup_second.dot(sup_op_second) 

        sup_echo = trf.kraus_to_super(ry_f(np.pi), my_hs_basis)

        total_sup_op_echo = sup_echo.dot(sup_op_second_tot.dot(sup_echo.dot(sup_op_first_tot)))

        
        if noise:
            res["F_pro_echo_noisy"] = \
                average_process_fidelity(cr_super_target, 
                                         total_sup_op_echo)
            res["F_gate_echo_noisy"] = \
                average_gate_fidelity(cr_super_target, 
                                      total_sup_op_echo, res["L1_noisy"])
        else:
            res["F_pro_echo"] = \
                average_process_fidelity(cr_super_target, 
                                         total_sup_op_echo)
            res["F_gate_echo"] = \
                average_gate_fidelity(cr_super_target, 
                                      total_sup_op_echo, res["L1"])
    return res 

def main():
    p_set_list = ["CR_1", "CR_2", "CR_3", "CR_4"]

    eps_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    result = {}

    start = time.time()
    
    for p_set in p_set_list:
        res = []
        for eps in eps_list:
            output = get_fidelity(eps, p_set)
            res.append(output)
        result[p_set] = res
    
    end = time.time()

    print("Computation time: {} s".format(end - start))
    
    save = True
    if save:
        with open("cr_fidelity_drive_.txt", "w") as fp:
            json.dump(result, fp)


if __name__ == '__main__':
    main()

    


    