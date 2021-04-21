from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Iterable, Union, Callable, List
from itertools import chain

import numpy as np
from scipy import linalg as la
import xarray as xr

from ..operators import id_op
from ..bases import OperatorBasis
from ..util.linalg import order_vecs, get_mat_elem, tensor_prod


class Qubit(ABC):
    def __init__(self,  label: str, basis: OperatorBasis):
        if not isinstance(basis, OperatorBasis):
            raise ValueError(
                "basis must be an instance of bases.OperatorBasis class")

        self._basis = basis

        if not isinstance(label, str):
            raise ValueError(
                "The qubit label must be a string type variable")
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new_label) -> None:
        self._label = new_label

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, new_basis):
        self._basis = new_basis

    @property
    def dim_hilbert(self):
        return self._basis.dim_hilbert

    @abstractmethod
    def hamiltonian(self) -> np.ndarray:
        pass

    @abstractmethod
    def potential(self) -> np.ndarray:
        pass

    @abstractmethod
    def wave_function(self) -> np.ndarray:
        pass

    @abstractmethod
    def _qubit_attrs(self) -> dict:
        pass

    @abstractmethod
    def dielectric_loss(self) -> List[np.ndarray]:
        pass

    def _get_eig_vals(self, subset_inds: Tuple[int]) -> np.ndarray:
        hamil = self.hamiltonian()
        eig_vals = la.eigh(
            hamil,
            eigvals_only=True,
            subset_by_index=subset_inds
        )
        return eig_vals

    def _get_eig_states(self, subset_inds: Tuple[int]) -> Tuple[np.ndarray, np.ndarray]:
        hamil = self.hamiltonian()
        eig_vals, eig_vecs = la.eigh(
            hamil,
            eigvals_only=False,
            subset_by_index=subset_inds
        )
        return eig_vals, eig_vecs.T

    def eig_energies(
        self,
        levels: Optional[Union[int, Iterable[int]]] = 10
    ) -> np.ndarray:
        if levels is not None:
            if isinstance(levels, int):
                if levels <= 1:
                    raise ValueError(
                        "Number of levels must be an integer greater than 1")
                subset_inds = (0, levels - 1)
                sel_inds = None
            elif isinstance(levels, Iterable):
                subset_inds = min(levels), max(levels)
                _inds = list(range(subset_inds[0], subset_inds[1] + 1))
                sel_inds = [_inds.index(level) for level in levels]
        else:
            subset_inds = None
            sel_inds = None

        eig_vals = self._get_eig_vals(subset_inds)
        if sel_inds:
            return order_vecs(eig_vals[sel_inds])
        return order_vecs(eig_vals)

    def eig_states(
            self,
            levels: Optional[Union[int, Iterable[int]]] = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        if levels is not None:
            if isinstance(levels, int):
                if levels <= 1:
                    raise ValueError(
                        "Number of levels must be an integer greater than 1")
                subset_inds = (0, levels - 1)
                sel_inds = None
            elif isinstance(levels, Iterable):
                subset_inds = min(levels), max(levels)
                _inds = list(range(subset_inds[0], subset_inds[1] + 1))
                sel_inds = [_inds.index(level) for level in levels]
        else:
            subset_inds = None
            sel_inds = None

        eig_vals, eig_vecs = self._get_eig_states(subset_inds)
        if sel_inds:
            return order_vecs(eig_vals[sel_inds], eig_vecs[sel_inds])
        return order_vecs(eig_vals, eig_vecs)

    def mat_elements(
        self,
        operator: Union[str, np.ndarray],
        levels: Union[int, Iterable[int]] = 10,
        *,
        as_xarray: Optional[bool] = False,
    ) -> np.ndarray:
        if isinstance(operator, str):
            if hasattr(self.basis, operator):
                _op = getattr(self.basis, operator)
                op = _op() if callable(_op) else _op
                if not isinstance(op, np.ndarray):
                    raise ValueError("Obtained operator is not a numpy array")
                if len(op.shape) != 2:
                    raise ValueError("The operator must be a 2D array")

            elif hasattr(self, operator):
                _op = getattr(self, operator)
                op = _op() if callable(_op) else _op
                if not isinstance(op, np.ndarray):
                    raise ValueError("Obtained operator is not a numpy array")
                if len(op.shape) != 2:
                    raise ValueError("The operator must be a 2D array")

            else:
                raise ValueError(
                    "Given operator string not supported by" +
                    "the qubit or its basis {}".format(str(self.basis)))
        elif isinstance(operator, np.ndarray):
            op = operator
        else:
            raise ValueError("Incorrect operator provided")

        _, in_states = self.eig_states(levels=levels)
        _, out_states = self.eig_states(levels=levels)

        mat_elems = get_mat_elem(op, in_states, out_states)

        if isinstance(levels, int):
            levels_arr = list(range(levels))
        elif isinstance(levels, Iterable):
            levels_arr = list(levels)

        if as_xarray:
            data_arr = xr.DataArray(
                data=mat_elems,
                dims=['in_levels', 'out_levels'],
                coords=dict(
                    in_levels=levels_arr,
                    out_levels=levels_arr,
                ),
                attrs=dict(
                    operator=op,
                    dim_hilbert=self.dim_hilbert,
                    basis=str(self.basis),
                    **self._qubit_attrs
                )
            )

            return data_arr
        return mat_elems

    def couple_to(
        self,
        other_qubit: 'Qubit',
        coupling: Callable = None,
        ** kwargs
    ) -> 'QubitSystem':
        if not isinstance(other_qubit, Qubit):
            raise ValueError(
                "Other qubit should be a Qubit object, "
                "instead got {}".format(type(other_qubit))
            )
        if self.label == other_qubit.label:
            raise ValueError("The two qubits must be uniquely labeled.")

        qubits = [self, other_qubit]

        if coupling is not None:
            if isinstance(coupling, Callable):
                if 'qubits' in kwargs:
                    raise ValueError(
                        "Multiple values for the keyword "
                        "arguement 'qubits' specified"
                    )
                return QubitSystem(qubits, coupling(qubits=qubits, **kwargs))
            else:
                raise ValueError(
                    "Coupling arguement expected to be pysqkit.CouplerTerm or "
                    "Callable, instead got {}".format(type(coupling))
                )
        else:
            return QubitSystem(qubits)


class Coupling:
    def __init__(
            self,
            prefactors: Union[float, complex, Iterable[Union[float, complex]]],
            operators: Iterable[Dict[str, np.ndarray]],
            qubits: Optional[List[str]] = None
    ):

        if isinstance(operators, dict):
            self._ops = [operators]
        elif isinstance(operators, Iterable):
            self._ops = list(operators)

        for op in self._ops:
            for key, val in op.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "Operator keys must be qubit labels of type str")
                if not isinstance(val, np.ndarray):
                    raise ValueError("The operators must be np.ndarray type")

        if qubits is None:
            qubit_set = set(chain.from_iterable(op.keys() for op in self._ops))
            self._qubits = sorted(list(qubit_set))
        else:
            involved_qubits = set(chain.from_iterable(op.keys()
                                                      for op in self._ops))

            qubit_set = set(qubits)
            if len(qubit_set) != len(qubits):
                raise ValueError("There are duplicate labels in the qubits")
            if qubit_set != involved_qubits:
                raise ValueError(
                    "Qubit labels have a mismatch with the operator labels")
            self._qubits = qubits

        self._hilbert_dims = {}
        for op in self._ops:
            for qubit, qubit_op in op.items():
                if len(qubit_op.shape) != 2 or qubit_op.shape[0] != qubit_op.shape[1]:
                    raise ValueError("Each operator must be a square matrix")

                if qubit in self._hilbert_dims:
                    if self._hilbert_dims[qubit] != qubit_op.shape[0]:
                        raise ValueError(
                            "Mismatch in the dimensionality of the qubit operators in different terms")
                else:
                    self._hilbert_dims[qubit] = qubit_op.shape[0]

        if isinstance(prefactors, (float, int, complex)):
            self._prefactors = [prefactors] * len(self._ops)
        elif isinstance(prefactors, Iterable):
            if len(prefactors) != len(self._ops):
                raise ValueError(
                    "Number of provided operators does not correspond to the number of terms in the Hamiltonian")
            self._prefactors = list(prefactors)

    @property
    def qubits(self):
        return self._qubits

    @property
    def hilbert_dims(self):
        return self._hilbert_dims

    @property
    def hilbert_dim(self):
        return np.prod(list(self._hilbert_dims.values()))

    def _get_hamiltonian(self) -> np.ndarray:
        dim = self.hilbert_dim
        hamiltonian = np.zeros((dim, dim), dtype='complex')

        for prefactor, op in self.hamiltonian_terms():
            hamiltonian += prefactor * op

        if np.all(np.real(hamiltonian)):
            return hamiltonian.real
        return hamiltonian

    def hamiltonian(self) -> np.ndarray:
        hamil = self._get_hamiltonian()
        return hamil

    def hamiltonian_terms(
            self,
            *,
            tensor_ops=True
    ) -> Iterable[Tuple[Union[float, complex], Union[np.ndarray, Dict[str, np.ndarray]]]]:
        if tensor_ops:
            for prefactor, term_ops in zip(self._prefactors, self._ops):
                inv_qs = list(term_ops.keys())
                total_ops = list(
                    term_ops[q] if q in inv_qs else np.eye(self.hilbert_dims[q]) for q in self._qubits
                )
                yield prefactor, tensor_prod(total_ops)
        else:
            for prefactor, term_ops in zip(self._prefactors, self._ops):
                yield prefactor, term_ops


class QubitSystem:
    def __init__(
        self,
        qubits: Iterable[Qubit],
        coupling: Optional[Union[Coupling, Iterable[Coupling]]] = None
    ) -> None:
        qubit_labels = []

        for qubit in qubits:
            if not isinstance(qubit, Qubit):
                raise ValueError(
                    "Each qubit must be a pysqkit.Qubit object, "
                    "instead got {}".format(type(qubit)))

            if qubit.label in qubit_labels:
                raise ValueError(
                    "Multiple qubits share the same label, "
                    "please ensure each qubit is uniquely labeled."
                )
            qubit_labels.append(qubit.label)

        self._qubits = list(qubits)
        self._labels = qubit_labels

        if coupling is not None:
            if not isinstance(coupling, Iterable):
                coupling = [coupling]
            for coup_term in coupling:
                if not isinstance(coup_term, Coupling):
                    raise ValueError(
                        "Each coupling should be "
                        "a pysqkit.Coupling "
                        " object, instead got {}".format(type(coup_term))
                    )

                for q in coup_term.qubits:
                    if q not in self._labels:
                        raise ValueError(
                            "Coupling terms involve qubit "
                            "{}, which is not part of the system".format(q))
            self._coupling = coupling
        else:
            self._coupling = []

    def __len__(self) -> int:
        return len(self._qubits)

    def __getitem__(self, q_label: Union[str, int]) -> Qubit:
        if isinstance(q_label, str):
            try:
                return self._qubits[self._labels.index(q_label)]
            except KeyError:
                raise KeyError(
                    "Qubit {} is not part of the state".format(q_label))
        elif isinstance(q_label, int):
            if q_label >= len(self._qubits) or q_label < 0:
                raise ValueError("Index outside of number of qubits in system")
            return self._qubits[q_label]

    def __contains__(self, qubit: Union[str, Qubit]) -> bool:
        if isinstance(qubit, str):
            return qubit in self._labels
        elif isinstance(qubit, Qubit):
            return qubit in self._qubits
        else:
            raise ValueError(
                "Can't check if a {} object is in system".format(type(qubit)))

    def __iter__(self) -> Iterable[Qubit]:
        yield from self._qubits

    @property
    def size(self):
        return len(self._qubits)

    @property
    def qubits(self) -> List[Qubit]:
        return self._qubits

    def index(self, qubit: Union[str, Qubit]) -> int:
        if isinstance(qubit, str):
            label = qubit
        elif isinstance(qubit, Qubit):
            label = qubit.label
        else:
            raise ValueError(
                "Provided qubit type must be a str or pysqkit.Qubit"
                "instead got {}".format(type(qubit))
            )
        if label in self._labels:
            return self._labels.index(label)
        else:
            raise ValueError("Qubit {} not in system".format(label))

    def bare_hamiltonian(
        self,
        *,
        truncated_levels: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        if truncated_levels:
            if not isinstance(truncated_levels, dict):
                raise ValueError(
                    "The truncated levels must be provided as a dictionary")

        if truncated_levels:
            subsys_dims = {
                q.label: truncated_levels[q.label] if q.label in \
                    truncated_levels else 6 for q in self.qubits}
        else:
            subsys_dims = {qubit.label: 6 for qubit in self.qubits}

        sys_dim = np.prod(list(subsys_dims.values()))

        bare_hamiltonian = np.zeros((sys_dim, sys_dim), dtype=complex)

        for subsys_ind, subsys_qubit in enumerate(self.qubits):
            eig_energies = subsys_qubit.eig_energies(
                levels=subsys_dims[subsys_qubit.label])
            q_hamil = np.diag(eig_energies)

            subsys_ops = [q_hamil if q.label == subsys_qubit.label else id_op(
                subsys_dims[q.label]) for q in self.qubits]
            subsys_hamil = tensor_prod(subsys_ops)

            bare_hamiltonian += subsys_hamil

        return bare_hamiltonian

    def int_hamiltonian(
        self,
        *,
        truncated_levels: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        if truncated_levels:
            if not isinstance(truncated_levels, dict):
                raise ValueError(
                    "The truncated levels must be provided as a dictionary")

        if truncated_levels:
            subsys_dims = {
                q.label: truncated_levels[q.label] if q in truncated_levels else 6 for q in self.qubits}
        else:
            subsys_dims = {qubit.label: 6 for qubit in self.qubits}

        sys_dim = np.prod(list(subsys_dims.values()))

        int_hamiltonian = np.zeros((sys_dim, sys_dim), dtype=complex)

        if self._coupling:
            for coup in self._coupling:
                for prefactor, term_ops in coup.hamiltonian_terms(tensor_ops=False):
                    coupled_qubits = list(term_ops.keys())

                    diag_ops = []
                    for qubit in self.qubits:
                        if qubit.label in coupled_qubits:
                            q_op = qubit.mat_elements(
                                term_ops[qubit.label],
                                levels=subsys_dims[qubit.label]
                            )
                            diag_ops.append(q_op)
                        else:
                            diag_ops.append(id_op(subsys_dims[qubit.label]))

                    op = tensor_prod(diag_ops)

                    int_hamiltonian += prefactor * op
        return int_hamiltonian

    def hamiltonian(
            self,
            *,
            truncated_levels: Optional[Dict[str, int]] = None
    ) -> np.ndarray:

        bare_hamiltonian = self.bare_hamiltonian(
            truncated_levels=truncated_levels
        )

        int_hamiltonian = self.int_hamiltonian(
            truncated_levels=truncated_levels
        )

        return bare_hamiltonian + int_hamiltonian

    def _get_eig_vals(
        self,
        subset_inds: Tuple[int],
        *,
        truncated_levels: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        hamil = self.hamiltonian(truncated_levels=truncated_levels)
        eig_vals = la.eigh(
            hamil,
            eigvals_only=True,
            subset_by_index=subset_inds
        )
        return eig_vals

    def _get_eig_states(
        self,
        subset_inds: Tuple[int],
        *,
        truncated_levels: Optional[Dict[str, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        hamil = self.hamiltonian(truncated_levels=truncated_levels)
        eig_vals, eig_vecs = la.eigh(
            hamil,
            eigvals_only=False,
            subset_by_index=subset_inds
        )
        return eig_vals, eig_vecs.T

    def eig_energies(
        self,
        levels: Optional[Union[int, Iterable[int]]] = 10,
        truncated_levels: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        if levels is not None:
            if isinstance(levels, int):
                if levels <= 1:
                    raise ValueError(
                        "Number of levels must be an integer greater than 1")
                subset_inds = (0, levels - 1)
                sel_inds = None
            elif isinstance(levels, Iterable):
                subset_inds = min(levels), max(levels)
                _inds = list(range(subset_inds[0], subset_inds[1] + 1))
                sel_inds = [_inds.index(level) for level in levels]
        else:
            subset_inds = None
            sel_inds = None

        eig_vals = self._get_eig_vals(
            subset_inds, truncated_levels=truncated_levels)
        if sel_inds:
            return order_vecs(eig_vals[sel_inds])
        return order_vecs(eig_vals)

    def eig_states(
            self,
            levels: Optional[Union[int, Iterable[int]]] = 10,
            truncated_levels: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if levels is not None:
            if isinstance(levels, int):
                if levels <= 1:
                    raise ValueError(
                        "Number of levels must be an integer greater than 1")
                subset_inds = (0, levels - 1)
                sel_inds = None
            elif isinstance(levels, Iterable):
                subset_inds = min(levels), max(levels)
                _inds = list(range(subset_inds[0], subset_inds[1] + 1))
                sel_inds = [_inds.index(level) for level in levels]
        else:
            subset_inds = None
            sel_inds = None

        eig_vals, eig_vecs = self._get_eig_states(
            subset_inds, truncated_levels=truncated_levels)
        if sel_inds:
            return order_vecs(eig_vals[sel_inds], eig_vecs[sel_inds])
        return order_vecs(eig_vals, eig_vecs)
    
    def convert_subsys_operator(
        self,
        label: str,
        op: np.ndarray,
        truncated_levels: Optional[Dict[str, int]] = None,
        as_qobj=False        
    ) -> np.ndarray:
        if label not in self._labels:
            raise ValueError("Unknown qubit label")

        if truncated_levels:
            if not isinstance(truncated_levels, dict):
                raise ValueError(
                    "The truncated levels must be provided as a dictionary")

        if truncated_levels:
            subsys_dims = {
                q.label: truncated_levels[q.label] if q.label in truncated_levels else 6 for q in self.qubits}
        else:
            subsys_dims = {qubit.label: 6 for qubit in self.qubits}

        sys_dim = np.prod(list(subsys_dims.values()))

        print(sys_dim)


        

        