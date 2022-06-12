# pysqkit

Pysqkit is an open-source Python package for studying superconducting qubits with a particular focus on the simulation of the dynamics of these systems.

Currently the focus is on transmon and fluxonium systems. For these qubits, Pysqkit allows tO extract the energy levels, wavefunctions and matrix elements.

Qubits can currently be coupled to each other, currently via capacitive couplers. A drive can be applied on each qubit of this system.

# Requirements
Most versions of the following packages should work. We also provide the explicit versions we used in our machine.

numpy (1.22.4)
scipy (3.5.2)
qutip (4.7.0)
matplotlib (3.5.2)
xarray (2022.3.0)

# Installation
Clone the pysqkit repository via git clone. Open the terminal and navigate to the parent folder where the pysqkit folder is stored and run

pip install pysqkit/

If you want your installation to be always up to date with the latest modifications run instead

pip install -e pysqkit/

# Documentation
The documentation is currently being written, we apologize for any inconvenience this causes. If you have questions regarding the code, you can always reach us via the github page.

# Contribute
If you would like to contribute to this project you are free to do so by either opening an issue for any bugs/problems or forking and creating a pull request for any other changes.

We expect contributions to comply with the PEP8 code style guidelines and we recommend the use of black for code formatting.

# License
This work is distributed under the Apache 2.0 License. See LICENSE.txt for the full conditions of the license.

# Citations
If you happen to use pysqkit for any research, please cite this project as:
TODO

# Additional infos

The code for the paper 

A. Ciani, B. Varbanov, N. Jolly, C. Andersen, B. Terhal "Microwave-activated gates between a fluxonium and a transmon qubit" (2022) 
can be found in the branch ciani_et_al in the folder ciani_et_al/
