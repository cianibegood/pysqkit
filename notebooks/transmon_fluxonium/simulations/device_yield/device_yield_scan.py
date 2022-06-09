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

# %%
from pathlib import Path

import numpy as np
import xarray as xr

from layouts import Layout
from layouts.library import surface_code
from layouts.util import (
    set_freq_groups, 
    set_transmon_target_freqs, 
    set_fluxonium_target_params, 
    sample_params, 
    any_collisions
)

# %%
NOTEBOOK_DIR = Path.cwd()

DATA_FOLDER = NOTEBOOK_DIR / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Scan with distance and resistance for fixed drive amplitude

# %%
_scan_version = 3

# %%
MIN_RES_VAR = 1e-3 # Minimum room-temp resistance variation coeff
MAX_RES_VAR = 1e-1 # Maximum room-temp resistance variation coeff
NUM_RES_VARS = 20 # Number of resistance variables

RES_VARS = np.geomspace(MIN_RES_VAR, MAX_RES_VAR, NUM_RES_VARS)

NUM_SEEDS = 6000 # Number of seeds to sample the collisions from
SEEDS = np.arange(1, NUM_SEEDS + 1)

DISTANCES = [3, 5, 7] # The surface code distance over which to scan

# The drive label which determined what set of collision bounds to use
DRIVE_STR = "low" # correspnding to drive amplitude of 100 MHz
#DRIVE_STR = "mid" # correspnding to drive amplitude of 300 MHz
#DRIVE_STR = "high" # correspnding to drive amplitude of 500 MHz

# %% [markdown]
# For the BOUNDS lists below: there are a total of 9 collision types that we have identified to be dominant.
#
# Collision type 2 does not a frequency bound associated with the collision (instead it defined an area). As such it is not included in the BOUNDS list.
#
# The collisions defined in the BOUNDS therefore correspod to:
# [type 1, type 3, type 4, type 5, type 6, type 7, type 8, type 9].

# %%
if DRIVE_STR == "low":
    # In this case collision type 6 is not bounded (drive is too weak to drive collision) - it is therefore set to 0.
    BOUNDS = [0.100, 0.015, 0.005, 0.009, 0.000, 0.005, 0.007, 0.010]
elif DRIVE_STR == "mid":
    # In this case collision type 5 is too close to collision type 4. We instead use the bound of type 4 for both and
    # instead set the bound on collision type 5 to 0
    BOUNDS = [0.100, 0.040, 0.040, 0, 0.017, 0.015, 0.020, 0.025]
elif DRIVE_STR == "high":
    # In this case collision type 5 is too close to collision type 4. We instead use the bound of type 4 for both and
    # instead set the bound on collision type 5 to 0
    BOUNDS = [0.100, 0.060, 0.050, 0, 0.035, 0.015, 0.020, 0.050]
else:
    raise ValueError("Unsupported drive strength label, must be either 'low', 'mid' or 'high'.")

# %%
GROUP_FREQS = np.array([4.3, 4.7, 5.3, 5.7]) # The four targeted transmon frequencies
GROUP_ANHARMS = np.repeat(-0.3, len(GROUP_FREQS)) # Each transmon has the same anharmonicity

CHARGE_ENERGY = 1 # The targeted charging energy
INDUCT_ENERGY = 1 # The targeted inductive energy
JOSEPH_ENERGY = 4 # The targeted josephson energy

# %%
for dist_ind, distance in enumerate(DISTANCES):
    result = np.zeros((NUM_RES_VARS, NUM_SEEDS), dtype=int)
    
    layout = surface_code(distance, mixed_layout=True)
    set_freq_groups(layout)

    set_transmon_target_freqs(layout, GROUP_FREQS, GROUP_ANHARMS)
    set_fluxonium_target_params(
        layout, 
        charge_energy = CHARGE_ENERGY, 
        induct_energy = INDUCT_ENERGY, 
        joseph_energy=  JOSEPH_ENERGY
    )

    for var_ind, res_var in enumerate(RES_VARS):
        for seed_ind, seed in enumerate(SEEDS):
            sample_params(layout, seed, res_var, num_junctions=100, num_fluxonium_levels=6)
            if not any_collisions(layout, BOUNDS):
                result[var_ind, seed_ind] = 1
                
    collision_arr = xr.DataArray(
        result,
        dims = ["resist_var", "seed"],
        coords = dict(
            distance = distance,
            resist_var = RES_VARS,
            seed = SEEDS
        )
    )
    
    data_arr_name = f"mixed_device_yield_d_{distance}_resist_{MIN_RES_VAR}_{MAX_RES_VAR}_seeds_{NUM_SEEDS}_v{_scan_version}_{DRIVE_STR}_drive.nc"
    collision_arr.to_netcdf(DATA_FOLDER / data_arr_name)
