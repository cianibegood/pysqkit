from typing import Callable, List, Union, Dict, Tuple

import qutip as qtp 
import numpy as np
import time

from pysqkit.solvers.solvkit import integrate
from pysqkit.systems.system import QubitSystem, Qubit, Drive
from pysqkit.util.linalg import hilbert_schmidt_prod
from pysqkit.util.hsbasis import iso_basis

from pysqkit.util.hsbasis import pauli_by_index

import multiprocessing
from functools import partial
         
class TomoEnv:   
    def __init__(
        self,
        system: Union[Qubit, QubitSystem],
        time: np.ndarray,
        options: qtp.solver.Options=None,
        with_noise: bool=False,
        dressed_noise: bool=False
        ):
            """
            Class to perform tomography
            """            
        
            self._system = system
            self._time = time
            self._options = options

            if isinstance(system, QubitSystem):
                q_dims = [qubit.dim_hilbert for qubit in system.qubits]
                self._dims_qobj = [q_dims, [1]*system.size]
                for qubit in self._system:
                        if qubit.is_driven:
                            for label, drive in qubit.drives.items():
                                drive.set_params(time=time/(2*np.pi))  

            elif isinstance(system, Qubit):
                q_dims = [system.dim_hilbert]
                self._dims_qobj = [q_dims, [1]]
                if system.is_driven:
                    for label, drive in system.drives.items():
                        drive.set_params(time=time/(2*np.pi))

            if with_noise:            
                if isinstance(system, QubitSystem):
                    if dressed_noise:
                        u = system.diagonalizing_unitary(as_qobj=True)
                        collapse_ops = [u*op*u.dag() for qubit in \
                            system for op in qubit.collapse_ops(as_qobj=True) \
                                if op is not None]
                        dephasing_ops = \
                            [u*qubit.dephasing_op(as_qobj=True)*u.dag() \
                                for qubit in system if \
                                    qubit.dephasing_op(as_qobj=True) is not None]
                    else:
                        collapse_ops = [op for qubit in \
                            system for op in qubit.collapse_ops(as_qobj=True)]
                        dephasing_ops = [qubit.dephasing_op(as_qobj=True) \
                            for qubit in system]
                    if None in dephasing_ops:
                        self._jump_op = collapse_ops
                    else:
                        self._jump_op = collapse_ops + dephasing_ops
                elif isinstance(system, Qubit):
                    collapse_ops = system.collapse_ops(as_qobj=True)
                    dephasing_op = system.dephasing_op(as_qobj=True)
                    if dephasing_op is None:
                        self._jump_op = collapse_ops
                    else:
                        self._jump_op = collapse_ops + [dephasing_op]
            else:
                self._jump_op = []

            self.hamil0 = self._system.hamiltonian(as_qobj=True)
            self.hamil_drive = []
            self.pulse_drive = []
            if isinstance(self._system, QubitSystem):
                for qubit in self._system:
                    if qubit.is_driven:
                        for label, drive in qubit.drives.items():
                            self.hamil_drive.append(drive.hamiltonian(as_qobj=True))
                            self.pulse_drive.append(drive.eval_pulse())
            elif isinstance(self._system, Qubit):
                if self._system.is_driven:
                    for label, drive in self._system.drives.items():
                        self.hamil_drive.append(drive.hamiltonian(as_qobj=True))
                        self.pulse_drive.append(drive.eval_pulse())
            
            self._hs_basis_speed_up = ["pauli_by_index"]
        
    @property
    def system(self):
        return self._system
    
    @property
    def time(self):
        return self._time
 

    def simulate(self, state_in):
              
        result = integrate(self._time, state_in, self.hamil0, self.hamil_drive,
                           self.pulse_drive, self._jump_op, 
                           "mesolve", options=self._options)
                    
        res = result.states[-1]
        return res  
    
    def evolve_hs_basis(
        self,
        i: int,
        input_states: List[np.ndarray],
        hs_basis: Callable[[int, int], np.ndarray]
    ) -> np.ndarray:
        
        """
        Description
        ----------------------------------------------------------------------
        Returns the action of the quantum operation associated
        with the time-evolution on the i-th element of a Hilbert-Schmidt
        basis defined via the function hs_basis with computational 
        basis states in input_states.
        """

        d = len(input_states)

        basis_i = hs_basis(i, d)
        eigvals, eigvecs = np.linalg.eig(basis_i)
        evolved_basis_i = 0

        for n in range(0, d):
            iso_eigvec = np.einsum('i,ij->j', eigvecs[:, n], input_states)
            iso_eigvec_qobj = qtp.Qobj(inpt=iso_eigvec, dims=self._dims_qobj)
            rho_iso_eigvec_qobj = iso_eigvec_qobj*iso_eigvec_qobj.dag()
            evolved_iso_eigvec = self.simulate(rho_iso_eigvec_qobj)
            evolved_basis_i += eigvals[n]*evolved_iso_eigvec[:, :]
        return evolved_basis_i
    
    def evolve_pauli_basis(
        self,
        i: int,
        input_states: List[np.ndarray]
    ):
        
        """
        Description
        ----------------------------------------------------------------------
        Returns the action of the quantum operation associated 
        with the time evolution on the i-th normalized Pauli operator. 
        It provides a speed-up over the more general evolve_hs_basis method
        that can be used with arbitrary Hilbert-Schmidt basis
        """

        if isinstance(self._system, QubitSystem):
            subsys_dims = list(q.dim_hilbert for q in self._system)
        elif isinstance(self._system, Qubit):
            subsys_dims = [self._system.dim_hilbert]

        d = len(input_states)

        pauli_op = np.sqrt(d)*pauli_by_index(i, d)

        proj_comp_subspace = np.einsum('ai, aj -> ij', input_states, 
                                       np.conj(input_states))
        pauli_op_repr = np.einsum('kl, ki, lj -> ij', pauli_op, 
                                  input_states, np.conj(input_states))
        
        if i == 0:
            rho_plus = qtp.Qobj(inpt=proj_comp_subspace/d, 
                                dims=[subsys_dims, subsys_dims], isherm=True)

            rho_plus_evolved = self.simulate(rho_plus)

            return np.sqrt(d)*rho_plus_evolved 
        
        else:
            # Mixed state on subspace with eigenvalue +1 

            rho_plus = qtp.Qobj(inpt=(proj_comp_subspace + pauli_op_repr)/(2*np.sqrt(d)), 
                                dims=[subsys_dims, subsys_dims], isherm=True)

            # Mixed state on subspace with eigenvalue -1 

            rho_minus = qtp.Qobj(inpt=(proj_comp_subspace - pauli_op_repr)/(2*np.sqrt(d)), 
                                dims=[subsys_dims, subsys_dims], isherm=True)

            rho_plus_evolved = self.simulate(rho_plus)
            rho_minus_evolved = self.simulate(rho_minus)

            return rho_plus_evolved - rho_minus_evolved
    
    def to_super( 
        self, 
        input_states: List[np.ndarray], 
        hs_basis: Callable[[int, int], np.ndarray],
        n_process: int=1,
        speed_up: bool=False
    ) -> np.ndarray:
    
        """
        Description
        ----------------------------------------------------------------------
        Returns the superoperator associated with the time-evolution 
        for states in input_states. The output_states are assumed to be 
        the same as the input states. The superoperator is written in the 
        Hilbert-Schmidt basis defined via the function hs_basis. 
        The function can be run in parallel by specifying the number of 
        processes n_process, which is 1 by default.

        """

        unsupported_basis = hs_basis.__name__ not in self._hs_basis_speed_up

        if speed_up == True and unsupported_basis:
            raise ValueError("Basis error: unsupported basis for speed-up"
                             "Set speed_up to False to run with this basis")

        d = len(input_states)
        superoperator = np.zeros([d**2, d**2], dtype=complex)
        
        basis = [iso_basis(i, input_states, hs_basis) for i in range(0, d**2)]
        index_list = np.arange(0, d**2)

        if n_process > 1:

            pool = multiprocessing.Pool(processes=n_process)
            if speed_up:
                func = partial(self.evolve_pauli_basis, 
                               input_states=input_states)
            else:
                func = partial(self.evolve_hs_basis, input_states=input_states, 
                               hs_basis=hs_basis)

            evolved_basis = pool.map(func, index_list, 
                                     chunksize=int(d**2//n_process))

            pool.close()
            pool.join()
        else:
            evolved_basis = []
            for index in index_list:
                if speed_up:
                    ev_tmp = self.evolve_pauli_basis(index, 
                                                     input_states=input_states)
                else:
                    ev_tmp = self.evolve_hs_basis(index, 
                                                  input_states=input_states, 
                                                  hs_basis=hs_basis)

                evolved_basis.append(ev_tmp)
        
        for i in range(0, d**2):
            for k in range(0, d**2):
                superoperator[k, i] = hilbert_schmidt_prod(basis[k], 
                                                           evolved_basis[i])
        
        return superoperator
    
    def leakage(
        self,
        input_states: List[np.ndarray]
    ) -> float:
        dim_subspace = len(input_states)
        _proj_comp = np.einsum('ai, aj -> ij', input_states, 
                               np.conj(input_states))
        
        if isinstance(self._system, QubitSystem):
            subsys_dims = list(q.dim_hilbert for q in self._system)
        elif isinstance(self._system, Qubit):
            subsys_dims = [self._system.dim_hilbert]

        proj_comp = qtp.Qobj(inpt=_proj_comp, 
                                 dims=[subsys_dims, subsys_dims], isherm=True)    
        res = self.simulate(proj_comp/dim_subspace)
        return 1 - qtp.expect(proj_comp, res)
    
    def seepage(
        self,
        input_states: List[np.ndarray]
    ) -> float:
        dim_subspace = len(input_states)
        _proj_comp = np.einsum('ai, aj -> ij', input_states, 
                               np.conj(input_states))
        subsys_dims = list(q.dim_hilbert for q in self._system)
        proj_comp = qtp.Qobj(inpt=_proj_comp, 
                             dims=[subsys_dims, subsys_dims], isherm=True)
        ide = qtp.Qobj(inpt=np.identity(self._system.dim_hilbert), 
                       dims=proj_comp.dims, isherm=True)
        proj_leak = ide - proj_comp
        dim_leak = self._system.dim_hilbert - dim_subspace
        res = self.simulate(proj_leak/dim_leak)
        return 1 - qtp.expect(proj_leak, res)

        
        


        

        
