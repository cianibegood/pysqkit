# Device yield and number of collisions scan notebooks.

By default running the notebook will save the results in the /data subdirectory (and will create it if it doesn't exist already).

The hidden variable _scan_version labels the various versions of the results as we discovered and included more collisions (or in one case forgot to include some).

The variables of the simulation are the normal temperature resistance variations, code distances, number of seeds and drive amplitude.

The choice of drive amplitude fixes each of the frequency collision bounds used in these simulations. There are three options supported: low (corresponding to 100 MHz), mid (300 MHz) and high (500 MHz). In the notebook all options are listed and 2 out of the 3 are uncommented. Choosing a drive amplitude should amount to uncommenting the chosen amplitude and uncommenting the other. The bounds are correspondingly chosen and correspond to what is defined in Table 2 of the manuscript (these are extracted from scans, see collision_bounds simulations and additional_plots). Note that for the number of collisions we only consider a single distance (in this case the mid strength drive).

For code distance we go over distance 3, 5, 7. Note that for the number of collisions we only consider a single distance (in this case 3).

We fix the number of seeds used in simulation to 6000, corresponding to the resample of the lattice frequencies. The seeds are used to initialize the random number generator and (hopefully) control the randomness of the simulation to ensure reproducibility. Note that the seeds start from 1 instead of 0.

We consider resistance variations (defined as the standard deviation of the resistance over the resistance value, making this unitless) in a given range, defined by MIN_RES_VAR and MAX_RES_VAR. We then take NUM_RES_VARS over this range using an equidistance spacing on a log scale (specifically using np.geomspace).

Other parameters in these simulations are the target transmon frequencies and anharmonicities as well as the target charging, inductive and josephson energy for each fluxonium (they are all ideally the same). These are defined following Table 1 in the manuscript.

To reproduce all the data used in the manuscript the only thing need is to select each of the possible drive amplitude labels and rerun the notebook each time.