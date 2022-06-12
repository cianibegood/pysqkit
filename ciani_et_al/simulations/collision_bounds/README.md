# Frequency collision bounds scan notebooks.

By default running the notebook will save the results in the /data subdirectory (and will create it if it doesn't exist already).

The target transmon and fluxonium parameters are stored in flx_transm_params.txt. In particular that files contains 4 sets of parameters used for the CR gate, one for each target transmon frequency. For the scans we choose one (the one at 5.3 GHz) and we believe the bounds for the other 3 frequencies will be similar.

The main parameter to these scans is the drive amplitude or the DRIVE_STR constant (this in turn fixes the global variable for the drive amplitude). There are three options supported: low (corresponding to 100 MHz), mid (300 MHz) and high (500 MHz). In the notebook all options are listed and 2 out of the 3 are uncommented. Choosing a drive amplitude should amount to uncommenting the chosen amplitude and uncommenting the other.

Each scan further defines a frequency range (FREQ_RANGE) around the collision over which to scan as well as the number of points (NUM_POINTS) to take in this range. These are defined for each frequency collision and the number of points is usually taken to be 41. Note that this makes the simulation fairly slow.

For collisions not involving a spectator only a 2-qubit system is simulated, while for the collisions involving a spectator the full 3-qubit system is simulated. When collisions involving a spectator qubit the parameters of the target transmons are further fixed and the spectator transmon frequency is scanned to identify the collisions. Again, we don't expected these to substantially differ for different target frequencies.

To reproduce all the data used in the manuscript the only thing need is to select each of the possible drive amplitude labels and rerun the notebook each time.