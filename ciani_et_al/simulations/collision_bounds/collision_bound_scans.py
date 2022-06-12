# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-resonance gate between a transmon and a fluxonium: driving the fluxonium
#
# In this notebook we study the cross-resonance two-qubit gate between a transmon and a fluxonium. Our goal is to extract the collisions that matter and perform detailed scans in their vicinity to extract the frequency collision bounds.

# %% [markdown]
# To extract the bound we look at the crosstalk or leakage and bound these to be below a certain level (or close to it, as bounds were selected by hand).

# %%
from itertools import product, combinations
from typing import List, Dict, Optional
import pathlib
import json
import cmath

import numpy as np
import xarray as xr
import qutip as qtp
from scipy.optimize import minimize
from scipy.integrate import simpson

from pysqkit import qubits, systems, couplers, drives

from pysqkit.drives.pulse_shapes import gaussian_top
from pysqkit.util.phys import temperature_to_thermalenergy
from pysqkit.util.linalg import get_mat_elem
from pysqkit.tomography import TomoEnv

from pysqkit.solvers.solvkit import integrate

# %%
SOLVER_OPTIONS = qtp.solver.Options()
SOLVER_OPTIONS.atol = 1e-12
SOLVER_OPTIONS.rtol = 1e-10


# %% [markdown]
# # Define auxillary functions used

# %%
def comp_state_labels(num_qubits: Optional[int] = 2):
    state_combinations = product("01", repeat=num_qubits)
    labels = ["".join(states) for states in state_combinations]
    return labels

def state_labels(*qubit_states):
    state_combinations = product(*qubit_states)
    labels = ["".join(states) for states in state_combinations]
    return labels

def get_states(system, state_labels):
    states = {}
    for label in state_labels:
        _state = system.state(label)[1]
        
        loc = np.argmax(np.abs(_state))
        phase = cmath.phase(_state[loc])
        
        states[label] = np.exp(-1j*phase) * _state
    return states


# %%
def extract_freqs(
    qubit: systems.Qubit, 
) -> Dict[str, float]:
    """
    Description
    --------------------------------------------------------------
    Returns the transition frequencies of a given qubit.
    """
    
    num_levels = qubit.dim_hilbert
    
    results = {}
    
    states = [qubit.state(str(level)) for level in range(num_levels)]
        
    for level_i, level_j in combinations(range(num_levels), 2):
        freq_i = states[level_i][0]
        freq_j = states[level_j][0]
        results[f"freq_{level_i}{level_j}"] = freq_j - freq_i

    return results


# %%
def zz_crosstalk(system: systems.QubitSystem) -> float:
    xi_zz = system.state('00')[0] + system.state('11')[0] \
        - system.state('01')[0] - system.state('10')[0]
    return xi_zz

def xz_coeff(comp_states, op) -> float:
    xz0 = get_mat_elem(op, comp_states['00'], comp_states['10'])
    xz1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(xz0 - xz1))/2

def xi_coeff(comp_states, op) -> float:
    xi0 = get_mat_elem(op, comp_states['00'], comp_states['10'] )
    xi1 = get_mat_elem(op, comp_states['01'], comp_states['11'] )
    return (np.imag(xi0 + xi1))/2


# %%
def func_to_minimize(
    x: List[float],
    rise_time: float,
    amp: float
) -> float:
    pulse_time = x[0]
    time_step = 1e-3
    num_points = int(pulse_time/time_step)
    times = np.linspace(0, pulse_time, num_points)
    
    pulse = gaussian_top(times, rise_time, pulse_time)
    integral = simpson(2*np.pi*amp*pulse, times)
    return np.abs(integral - np.pi/4)  #Watch out factor of 2?  


# %%
def minimize_drive_time(
    system: systems.QubitSystem,
    comp_states: List[np.ndarray],
    eps_drive : float,
    rise_time : Optional[float] = 5.0,
    *,
    init_time : Optional[float] = 100.0,
) -> float:
    
    q_op = system["control"].charge_op()
    args_to_pass = (rise_time, np.abs(xz_coeff(comp_states, q_op))*0.5*eps_drive)

    try:
        minimization_result = minimize(
            func_to_minimize, 
            init_time,
            args=args_to_pass
        )
        gate_time = minimization_result['x'][0]
    except ValueError:
        gate_time = None
    
    return gate_time

def get_drive_params(
    system: systems.QubitSystem,
    comp_states: List[np.ndarray],
    eps_drive : float,
    rise_time : Optional[float] = 10.0,
    *,
    init_time : Optional[float] = 20.0,
    detuning : Optional[float] = 0,
    points_per_period : Optional[int] = 10
):
    pulse_time = minimize_drive_time(
        system = coupled_sys,
        comp_states = comp_states,
        eps_drive = eps_drive,
        rise_time = rise_time,
        init_time=init_time
    )
    
    if pulse_time is None:
        raise ValueError("Pulse time minimization has failed.") # Should be handled differently
    
    drive_freq = system["target"].freq
    
    num_points = int(pulse_time * drive_freq * points_per_period)
    times = np.linspace(0, pulse_time, num_points)

    params = dict(
        phase = 0, 
        time = times, 
        rise_time = rise_time, 
        pulse_time = pulse_time,
        amp = eps_drive, 
        freq = drive_freq + detuning
    )
    return params


# %%
def run_simulation(
    times: np.ndarray,
    system: systems.QubitSystem, 
    init_state: qtp.qobj.Qobj, 
    options: Optional[qtp.solver.Options] = None,
    *,
    solver="mesolve",
    with_noise: Optional[bool] = False
) -> qtp.solver.Result:
    sys_hamil = system.hamiltonian(as_qobj=True)
    drive_hamils = []
    drive_pulses = []
                    
    for qubit in system:
        if qubit.is_driven:
            for label, drive in qubit.drives.items():
                drive_hamils.append(drive.hamiltonian(as_qobj=True))
                drive_pulses.append(drive.eval_pulse())
    
    if with_noise:
        jump_ops = [op for qubit in system for op in qubit.collapse_ops(as_qobj=True)]
    else:
        jump_ops = []
                    
    result = integrate(
        times, 
        init_state, 
        sys_hamil, 
        drive_hamils,
        drive_pulses, 
        jump_ops, 
        solver=solver, 
        options=options
    )
                    
    return result  


# %%
def get_probabilities(
    state_labels : List[str], 
    system : systems.QubitSystem, 
    output_states : List[qtp.Qobj],
) -> Dict:
    probs_dict = {}
    
    for label in state_labels:
        probs = []
        
        state = system.state(label, as_qobj=True)[1]        
        projector = state*state.dag()
        
        for out_state in output_states:
            prob = qtp.expect(projector, out_state)
            probs.append(prob)
        
        probs_dict[label] = probs
    return probs_dict


# %%
def get_leakage(
    times: np.ndarray,
    system: systems.QubitSystem,
    comp_states: List[np.ndarray],
    options: Optional[qtp.solver.Options] = None,
    *,
    with_noise: Optional[bool] = False
):
    env_syst = TomoEnv(
        system=system, 
        time=2*np.pi*times, 
        options=options, 
        with_noise=with_noise
    )
    
    leakage = env_syst.leakage(comp_states)
    return leakage    

# %% [markdown]
# # Set up directories for data and image saving

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()

DATA_FOLDER = NOTEBOOK_DIR / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# %% [markdown] heading_collapsed=true
# # Introducing the qubits and the coupled system

# %%
with open(NOTEBOOK_DIR / "flx_transm_params.txt") as param_file:
    PARAM_SETS = json.load(param_file)
    
SET_LABEL = "CR_3"
PARAM_SET = PARAM_SETS[SET_LABEL]

# %% hidden=true
TEMPERATURE = 0.020 # Environment temperature, 20mK
THERMAL_ENERGY = temperature_to_thermalenergy(TEMPERATURE)

TRANSMON_LEVELS = 3
FLUXONIUM_LEVELS = 6

# We will save a few parameters as constants that we'll reuse throughout the simulations
# This mostly makes the code cleaner
TARGET_FREQ = PARAM_SET["max_freq_t"]
TRANSMON_ANHARM = PARAM_SET["anharm_t"]
DIEL_LOSS_TANGENT = PARAM_SET["diel_loss_tan_t"]

COUP_STRENGTH = PARAM_SET["jc"]

# %%
#Control fluxonium
control_fluxonium = qubits.Fluxonium(
    label = 'control', 
    charge_energy = PARAM_SET["charge_energy_f"], 
    induct_energy = PARAM_SET["induct_energy_f"], 
    joseph_energy = PARAM_SET["joseph_energy_f"], 
    diel_loss_tan = PARAM_SET["diel_loss_tan_f"],
    env_thermal_energy = THERMAL_ENERGY,
)
control_fluxonium.diagonalize_basis(FLUXONIUM_LEVELS)

# We also add the CR drive on the fluxonium (parameters specified later)
control_fluxonium.add_drive(
    drives.microwave_drive,
    label = 'cr_drive',
    pulse = drives.pulses.cos_modulation,
    pulse_shape = drives.pulse_shapes.gaussian_top
)

# %%
fluxonium_freqs = extract_freqs(control_fluxonium)

# %%
# The drive label which determined what set of collision bounds to use
DRIVE_STR = "low" # correspnding to drive amplitude of 100 MHz
#DRIVE_STR = "mid" # correspnding to drive amplitude of 300 MHz
#DRIVE_STR = "high" # correspnding to drive amplitude of 500 MHz

# %%
if DRIVE_STR == "low":
    EPS_DRIVE = 0.1 # GHz
elif DRIVE_STR == "mid":
    EPS_DRIVE = 0.3 # GHz
elif DRIVE_STR == "high":
    EPS_DRIVE = 0.5 # GHz
else:
    raise ValueError("Unsupported drive strength label, must be either 'low', 'mid' or 'high'.")
    
RISE_TIME = 10

# %%
LATTICE_TYPE = "mixed"

# %% [markdown]
# # Scans around collisions leading to high ZZ crosstalk

# %% [markdown]
# ### Scan around the $f_{control}^{1 \rightarrow 2} = f_{target}^{0 \rightarrow 1}$ frequency collision (Type 1)

# %%
SAVE_DATA = True
FREQ_RANGE = 0.15 #GHz
NUM_POINTS = 41

collision_cond = fluxonium_freqs["freq_12"]
target_freqs = np.linspace(collision_cond - FREQ_RANGE, collision_cond + FREQ_RANGE, NUM_POINTS)

results = []
comp_labels = comp_state_labels(2)

for tar_freq in target_freqs:
    target_tmon = qubits.SimpleTransmon(
        label = 'target', 
        max_freq = tar_freq, 
        anharm = TRANSMON_ANHARM,
        diel_loss_tan = DIEL_LOSS_TANGENT,
        env_thermal_energy = THERMAL_ENERGY,    
        dim_hilbert = TRANSMON_LEVELS
    )
    
    coupled_sys = target_tmon.couple_to(
        control_fluxonium, 
        coupling = couplers.capacitive_coupling, 
        strength=COUP_STRENGTH,
    )
    
    results.append(zz_crosstalk(coupled_sys))
    
zz_crosstalks = xr.DataArray(
    results,
    dims = ["target_freq"],
    coords = dict(target_freq = target_freqs),
    attrs = dict(
        fluxonium_charge_energy = control_fluxonium.charge_energy,
        fluxonium_induct_energy = control_fluxonium.induct_energy,
        fluxonium_joseph_energy = control_fluxonium.joseph_energy,
        anharm = TRANSMON_ANHARM,
        collision_cond = collision_cond,
        points_per_drive_period = 10,
        transmon_levels = TRANSMON_LEVELS,
        fluxonium_levels = FLUXONIUM_LEVELS,
    )
)

if SAVE_DATA:
    collision_type = "zz_crosstalk"
    trans = "ctrl_12"    
    da_name = f"{LATTICE_TYPE}_lat_{collision_type}_col_{trans}_transition_{DRIVE_STR}_drive_scan.nc"
    zz_crosstalks.to_netcdf(DATA_FOLDER / da_name)

# %% [markdown]
# ### Scan around the $f_{control}^{0 \rightarrow 3} = f_{target}^{0 \rightarrow 1}$ frequency collision (Type 1)

# %%
SAVE_DATA = True
FREQ_RANGE = 0.15 #GHz
NUM_POINTS = 41

collision_cond = fluxonium_freqs["freq_03"]
target_freqs = np.linspace(collision_cond - FREQ_RANGE, collision_cond + FREQ_RANGE, NUM_POINTS)

results = []
comp_labels = comp_state_labels(2)

for tar_freq in target_freqs:
    target_tmon = qubits.SimpleTransmon(
        label = 'target', 
        max_freq = tar_freq, 
        anharm = TRANSMON_ANHARM,
        diel_loss_tan = DIEL_LOSS_TANGENT,
        env_thermal_energy = THERMAL_ENERGY,    
        dim_hilbert = TRANSMON_LEVELS
    )
    
    coupled_sys = target_tmon.couple_to(
        control_fluxonium, 
        coupling = couplers.capacitive_coupling, 
        strength=COUP_STRENGTH,
    )
    
    results.append(zz_crosstalk(coupled_sys))
    
zz_crosstalks = xr.DataArray(
    results,
    dims = ["target_freq"],
    coords = dict(target_freq = target_freqs),
    attrs = dict(
        fluxonium_charge_energy = control_fluxonium.charge_energy,
        fluxonium_induct_energy = control_fluxonium.induct_energy,
        fluxonium_joseph_energy = control_fluxonium.joseph_energy,
        anharm = TRANSMON_ANHARM,
        collision_cond = collision_cond,
        points_per_drive_period = 10,
        transmon_levels = TRANSMON_LEVELS,
        fluxonium_levels = FLUXONIUM_LEVELS,
    )
)

if SAVE_DATA:
    collision_type = "zz_crosstalk"
    trans = "ctrl_03"    
    da_name = f"{LATTICE_TYPE}_lat_{collision_type}_col_{trans}_transition_{DRIVE_STR}_drive_scan.nc"
    zz_crosstalks.to_netcdf(DATA_FOLDER / da_name)

# %% [markdown]
# # Scans around frequency collisions involved in the CR gate between two qubits

# %% [markdown]
# ### Scan around the $f_{control}^{0 \rightarrow 4} = 2f_{target}^{0 \rightarrow 1}$ frequency collision (Type 3)

# %%
SAVE_DATA = True
FREQ_RANGE = 0.15 #GHz
NUM_POINTS = 41

collision_cond = 0.5*fluxonium_freqs["freq_04"]
target_freqs = np.linspace(collision_cond - FREQ_RANGE, collision_cond + FREQ_RANGE, NUM_POINTS)

results = []
comp_labels = comp_state_labels(2)

for tar_freq in target_freqs:
    target_tmon = qubits.SimpleTransmon(
        label = 'target', 
        max_freq = tar_freq, 
        anharm = TRANSMON_ANHARM,
        diel_loss_tan = DIEL_LOSS_TANGENT,
        env_thermal_energy = THERMAL_ENERGY,    
        dim_hilbert = TRANSMON_LEVELS
    )
    
    coupled_sys = target_tmon.couple_to(
        control_fluxonium, 
        coupling = couplers.capacitive_coupling, 
        strength=COUP_STRENGTH,
    )
    
    comp_states = get_states(coupled_sys, comp_labels)
    
    drive_params = get_drive_params(
        system = coupled_sys,
        comp_states = comp_states,
        eps_drive = EPS_DRIVE,
        rise_time = RISE_TIME,
    )
    times = drive_params["time"]
    coupled_sys['control'].drives['cr_drive'].set_params(**drive_params)
    
    comp_states_list = list(comp_states.values())
    
    leakage_rate = get_leakage(
        times = times,
        system = coupled_sys,
        comp_states = comp_states_list,
        options = SOLVER_OPTIONS,
        with_noise = False
    )

    results.append(leakage_rate)
    
leakage_rates = xr.DataArray(
    results,
    dims = ["target_freq"],
    coords = dict(target_freq = target_freqs),
    attrs = dict(
        fluxonium_charge_energy = control_fluxonium.charge_energy,
        fluxonium_induct_energy = control_fluxonium.induct_energy,
        fluxonium_joseph_energy = control_fluxonium.joseph_energy,
        anharm = TRANSMON_ANHARM,
        collision_cond = collision_cond,
        points_per_drive_period = 10,
        rise_time = RISE_TIME,
        eps_drive = EPS_DRIVE,
        transmon_levels = TRANSMON_LEVELS,
        fluxonium_levels = FLUXONIUM_LEVELS,
    )
)

if SAVE_DATA:
    collision_type = "cross_res"
    n_photons = 2
    trans = "ctrl_04"
    da_name = f"{LATTICE_TYPE}_lat_{collision_type}_col_{trans}_{n_photons}-photon_transition_{DRIVE_STR}_drive_scan.nc"
    leakage_rates.to_netcdf(DATA_FOLDER / da_name)

# %% [markdown]
# ### Scan around the $f_{control}^{0 \rightarrow 5} = 3f_{target}^{0 \rightarrow 1}$ frequency collision (Type 6)

# %%
SAVE_DATA = True
FREQ_RANGE = 0.1 #GHz
NUM_POINTS = 41

collision_cond = fluxonium_freqs["freq_05"]/3
target_freqs = np.linspace(collision_cond - FREQ_RANGE, collision_cond + FREQ_RANGE, NUM_POINTS)

results = []
comp_labels = comp_state_labels(2)

for tar_freq in target_freqs:
    target_tmon = qubits.SimpleTransmon(
        label = 'target', 
        max_freq = tar_freq, 
        anharm = TRANSMON_ANHARM,
        diel_loss_tan = DIEL_LOSS_TANGENT,
        env_thermal_energy = THERMAL_ENERGY,    
        dim_hilbert = TRANSMON_LEVELS
    )
    
    coupled_sys = target_tmon.couple_to(
        control_fluxonium, 
        coupling = couplers.capacitive_coupling, 
        strength=COUP_STRENGTH,
    )
    
    comp_states = get_states(coupled_sys, comp_labels)
    
    drive_params = get_drive_params(
        system = coupled_sys,
        comp_states = comp_states,
        eps_drive = EPS_DRIVE,
        rise_time = RISE_TIME,
    )
    times =  drive_params["time"]
    coupled_sys['control'].drives['cr_drive'].set_params(**drive_params)
    
    comp_states_list = list(comp_states.values())
    
    leakage_rate = get_leakage(
        times = times,
        system = coupled_sys,
        comp_states = comp_states_list,
        options = SOLVER_OPTIONS,
        with_noise = False
    )

    results.append(leakage_rate)
    
leakage_rates = xr.DataArray(
    results,
    dims = ["target_freq"],
    coords = dict(target_freq = target_freqs),
    attrs = dict(
        fluxonium_charge_energy = control_fluxonium.charge_energy,
        fluxonium_induct_energy = control_fluxonium.induct_energy,
        fluxonium_joseph_energy = control_fluxonium.joseph_energy,
        anharm = TRANSMON_ANHARM,
        collision_cond = collision_cond,
        points_per_drive_period = 10,
        rise_time = RISE_TIME,
        eps_drive = EPS_DRIVE,
        transmon_levels = TRANSMON_LEVELS,
        fluxonium_levels = FLUXONIUM_LEVELS,
    )
)

if SAVE_DATA:
    collision_type = "cross_res"
    n_photons = 3
    trans = "ctrl_05"
    da_name = f"{LATTICE_TYPE}_lat_{collision_type}_col_{trans}_{n_photons}-photon_transition_{DRIVE_STR}_drive_scan.nc"
    leakage_rates.to_netcdf(DATA_FOLDER / da_name)

# %% [markdown]
# ### Scan around the $f_{control}^{1 \rightarrow 5} = 2f_{target}^{0 \rightarrow 1}$ frequency collision (Type 4 and Type 5)

# %%
SAVE_DATA = True
FREQ_RANGE = 0.1 #GHz
NUM_POINTS = 41

collision_cond = fluxonium_freqs["freq_15"]/2
target_freqs = np.linspace(collision_cond - FREQ_RANGE, collision_cond + FREQ_RANGE, NUM_POINTS)

results = []
comp_labels = comp_state_labels(2)

for tar_freq in target_freqs:
    target_tmon = qubits.SimpleTransmon(
        label = 'target', 
        max_freq = tar_freq, 
        anharm = TRANSMON_ANHARM,
        diel_loss_tan = DIEL_LOSS_TANGENT,
        env_thermal_energy = THERMAL_ENERGY,    
        dim_hilbert = TRANSMON_LEVELS
    )
    
    coupled_sys = target_tmon.couple_to(
        control_fluxonium, 
        coupling = couplers.capacitive_coupling, 
        strength=COUP_STRENGTH,
    )
    
    comp_states = get_states(coupled_sys, comp_labels)
    
    drive_params = get_drive_params(
        system = coupled_sys,
        comp_states = comp_states,
        eps_drive = EPS_DRIVE,
        rise_time = RISE_TIME,
    )
    times =  drive_params["time"]
    coupled_sys['control'].drives['cr_drive'].set_params(**drive_params)
    
    comp_states_list = list(comp_states.values())
    
    leakage_rate = get_leakage(
        times = times,
        system = coupled_sys,
        comp_states = comp_states_list,
        options = SOLVER_OPTIONS,
        with_noise = False
    )

    results.append(leakage_rate)
    
leakage_rates = xr.DataArray(
    results,
    dims = ["target_freq"],
    coords = dict(target_freq = target_freqs),
    attrs = dict(
        fluxonium_charge_energy = control_fluxonium.charge_energy,
        fluxonium_induct_energy = control_fluxonium.induct_energy,
        fluxonium_joseph_energy = control_fluxonium.joseph_energy,
        anharm = TRANSMON_ANHARM,
        collision_cond = collision_cond,
        points_per_drive_period = 10,
        rise_time = RISE_TIME,
        eps_drive = EPS_DRIVE,
        transmon_levels = TRANSMON_LEVELS,
        fluxonium_levels = FLUXONIUM_LEVELS,
    )
)

if SAVE_DATA:
    collision_type = "cross_res"
    n_photons = 2
    trans = "ctrl_15"
    da_name = f"{LATTICE_TYPE}_lat_{collision_type}_col_{trans}_{n_photons}-photon_transition_{DRIVE_STR}_drive_scan.nc"
    leakage_rates.to_netcdf(DATA_FOLDER / da_name)

# %% [markdown]
# # Scans around frequency collisions involving a spectator transmon

# %% [markdown]
# We will first set up the fluxonium and target transmon, and scan over the spectator transmon frequency. For this we will define the traget transmon and the coupling to the fluxonium. We will also optimize the drive time for the given drive amplitude (since this isn't expected to change a lot as a function of the spectator frequency).

# %%
target_transmon = qubits.SimpleTransmon(
    label = 'target', 
    max_freq = TARGET_FREQ, 
    anharm = TRANSMON_ANHARM,
    diel_loss_tan = DIEL_LOSS_TANGENT,
    env_thermal_energy = THERMAL_ENERGY,    
    dim_hilbert = TRANSMON_LEVELS
)

coupled_sys = target_transmon.couple_to(
    control_fluxonium, 
    coupling = couplers.capacitive_coupling, 
    strength=COUP_STRENGTH,
)

target_control_coup = couplers.capacitive_coupling(
    qubits=[target_transmon, control_fluxonium],
    strength=COUP_STRENGTH,
)

comp_labels = comp_state_labels(2)
comp_states = get_states(coupled_sys, comp_labels)

drive_params = get_drive_params(
    system = coupled_sys,
    comp_states = comp_states,
    eps_drive = EPS_DRIVE,
    rise_time = RISE_TIME,
)
times =  drive_params["time"]

# %% [markdown]
# ### Scan around the $f_{spec}^{0 \rightarrow 1} + f_{target}^{0\rightarrow 1} = f_{control}^{0\rightarrow 4}$ frequency collision (Type 9)

# %%
SAVE_DATA = True
FREQ_RANGE = 0.1 #GHz
NUM_POINTS = 41

collision_cond = fluxonium_freqs["freq_04"] - target_transmon.freq
spectator_freqs = np.linspace(collision_cond - FREQ_RANGE, collision_cond + FREQ_RANGE, NUM_POINTS)

results = []
comp_labels = comp_state_labels(3)

for spec_freq in spectator_freqs:
    spec_transmon = qubits.SimpleTransmon(
        label = 'spectator', 
        max_freq = spec_freq, 
        anharm = TRANSMON_ANHARM,
        diel_loss_tan = DIEL_LOSS_TANGENT,
        env_thermal_energy = THERMAL_ENERGY,    
        dim_hilbert = TRANSMON_LEVELS
    )
    
    control_spectator_coup = couplers.capacitive_coupling(
        qubits=[control_fluxonium, spec_transmon],
        strength=COUP_STRENGTH,
    )
    
    full_sys = systems.QubitSystem(
        qubits=[target_transmon, control_fluxonium, spec_transmon],
        coupling=[target_control_coup, control_spectator_coup]
    )
    
    full_sys['control'].drives['cr_drive'].set_params(**drive_params)
    
    comp_states = get_states(full_sys, comp_labels)
    comp_states_list = list(comp_states.values())
    
    leakage_rate = get_leakage(
        times = times,
        system = full_sys,
        comp_states = comp_states_list,
        options = SOLVER_OPTIONS,
        with_noise = False
    )
    
    results.append(leakage_rate)
    
leakage_rates = xr.DataArray(
    results,
    dims = ["spectator_freq"],
    coords = dict(spectator_freq = spectator_freqs),
    attrs = dict(
        fluxonium_charge_energy = control_fluxonium.charge_energy,
        fluxonium_induct_energy = control_fluxonium.induct_energy,
        fluxonium_joseph_energy = control_fluxonium.joseph_energy,
        target_freq = target_transmon.freq,
        anharm = TRANSMON_ANHARM,
        collision_cond = collision_cond,
        rise_time = RISE_TIME,
        eps_drive = EPS_DRIVE,
        pulse_time = drive_params["pulse_time"],
        drive_freq = drive_params["freq"],
        points_per_drive_period = 10,
        transmon_levels = TRANSMON_LEVELS,
        fluxonium_levels = FLUXONIUM_LEVELS,
    )
)

if SAVE_DATA:
    collision_type = "spectator"
    n_photons = 2
    trans = "ctrl_04"
    da_name = f"{LATTICE_TYPE}_lat_{collision_type}_col_{trans}_{n_photons}-photon_transition_{DRIVE_STR}_drive_scan.nc"
    leakage_rates.to_netcdf(DATA_FOLDER / da_name)

# %% [markdown]
# ### Scan around the $f_{spec}^{1 \rightarrow 2} = f_{target}^{0\rightarrow1}$ frequency collision (Type 8)

# %%
SAVE_DATA = True
FREQ_RANGE = 0.1 #GHz
NUM_POINTS = 41

collision_cond = target_transmon.freq - TRANSMON_ANHARM
spectator_freqs = np.linspace(collision_cond - FREQ_RANGE, collision_cond + FREQ_RANGE, NUM_POINTS)

results = []
comp_labels = comp_state_labels(3)

for spec_freq in spectator_freqs:
    spec_transmon = qubits.SimpleTransmon(
        label = 'spectator', 
        max_freq = spec_freq, 
        anharm = TRANSMON_ANHARM,
        diel_loss_tan = DIEL_LOSS_TANGENT,
        env_thermal_energy = THERMAL_ENERGY,    
        dim_hilbert = TRANSMON_LEVELS
    )
    
    control_spectator_coup = couplers.capacitive_coupling(
        qubits=[control_fluxonium, spec_transmon],
        strength=COUP_STRENGTH,
    )
    
    full_sys = systems.QubitSystem(
        qubits=[target_transmon, control_fluxonium, spec_transmon],
        coupling=[target_control_coup, control_spectator_coup]
    )
    
    full_sys['control'].drives['cr_drive'].set_params(**drive_params)
    
    comp_states = get_states(full_sys, comp_labels)
    comp_states_list = list(comp_states.values())
    
    leakage_rate = get_leakage(
        times = times,
        system = full_sys,
        comp_states = comp_states_list,
        options = SOLVER_OPTIONS,
        with_noise = False
    )
    
    results.append(leakage_rate)
    
leakage_rates = xr.DataArray(
    results,
    dims = ["spectator_freq"],
    coords = dict(spectator_freq = spectator_freqs),
    attrs = dict(
        fluxonium_charge_energy = control_fluxonium.charge_energy,
        fluxonium_induct_energy = control_fluxonium.induct_energy,
        fluxonium_joseph_energy = control_fluxonium.joseph_energy,
        target_freq = target_transmon.freq,
        anharm = TRANSMON_ANHARM,
        collision_cond = collision_cond,
        rise_time = RISE_TIME,
        eps_drive = EPS_DRIVE,
        pulse_time = drive_params["pulse_time"],
        drive_freq = drive_params["freq"],
        points_per_drive_period = 10,
        transmon_levels = TRANSMON_LEVELS,
        fluxonium_levels = FLUXONIUM_LEVELS,
    )
)

if SAVE_DATA:
    collision_type = "spectator"
    n_photons = 1
    trans = "spec_12"
    da_name = f"{LATTICE_TYPE}_lat_{collision_type}_col_{trans}_{n_photons}-photon_transition_{DRIVE_STR}_drive_scan.nc"
    leakage_rates.to_netcdf(DATA_FOLDER / da_name)

# %%
