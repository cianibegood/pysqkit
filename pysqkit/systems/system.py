from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Iterable, Union, Callable, List
from itertools import chain, product 
from copy import copy
from functools import reduce
from inspect import signature

import numpy as np
from scipy import linalg as la
import xarray as xr
from qutip import Qobj

from ..bases import OperatorBasis
from ..util.linalg import order_vecs, get_mat_elem, tensor_prod


class Qubit(ABC):
    def __init__(
        self, 
        label: str, 
        basis: Union[OperatorBasis, Dict[str, OperatorBasis]]
        ):
        if not isinstance(basis, OperatorBasis) and \
            not isinstance(basis, Dict):
            raise ValueError("basis must be an instance of bases.OperatorBasis class"
                             " or an instance of Dict[str, bases.OperatorBasis]")

        self._basis = basis

        if not isinstance(label, str):
            raise ValueError("The qubit label must be a string type variable")
        self._label = label
        self._drives = {}

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
    def dim_hilbert(self) -> int:
        return self._basis.truncated_dim

    @property
    def drives(self) -> Dict[str, "Drive"]:
        return self._drives

    @property
    def is_driven(self) -> bool:
        return len(self.drives) > 0

    @abstractmethod
    def hamiltonian(self, *, expand_op=True) -> np.ndarray:
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
    def collapse_ops(self):
        pass 

    @abstractmethod
    def dephasing_op(self):
        pass 

    def _get_eig_vals(
        self,
        subset_inds: Tuple[int],
        expand: bool,
    ) -> np.ndarray:
        hamil = self.hamiltonian(expand=expand)
        eig_vals = la.eigh(hamil, eigvals_only=True, subset_by_index=subset_inds)
        return eig_vals

    def _get_eig_states(
        self,
        subset_inds: Tuple[int],
        expand: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        hamil = self.hamiltonian(expand=expand)
        eig_vals, eig_vecs = la.eigh(
            hamil, eigvals_only=False, subset_by_index=subset_inds
        )
        return eig_vals, eig_vecs.T

    def eig_energies(
        self,
        levels: Optional[Union[int, Iterable[int]]] = None,
        *,
        expand: Optional[bool] = True,
    ) -> np.ndarray:
        subset_inds, sel_inds = self._parse_levels(levels)
        eig_vals = self._get_eig_vals(subset_inds, expand)
        if sel_inds is not None:
            return order_vecs(eig_vals[sel_inds])
        return order_vecs(eig_vals)

    def eig_states(
        self,
        levels: Optional[Union[int, Iterable[int]]] = None,
        *,
        expand: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        subset_inds, sel_inds = self._parse_levels(levels)
        eig_vals, eig_vecs = self._get_eig_states(subset_inds, expand)
        if sel_inds is not None:
            return order_vecs(eig_vals[sel_inds], eig_vecs[sel_inds])
        return order_vecs(eig_vals, eig_vecs)

    def diagonalize_basis(self, num_levels: int) -> None:
        if not isinstance(num_levels, int):
            raise ValueError("Number of levels must be an integer")
        if num_levels < 1 or num_levels > self.dim_hilbert:
            raise ValueError(
                "The number of level must be between 1 "
                "and the dimensionality of the "
                "system d={}".format(self.dim_hilbert)
            )

        _, eig_states = self.eig_states(levels=self.dim_hilbert, expand=False)
        self.basis.transform(eig_states)
        self.basis.truncate(num_levels)
    
    def state(
        self,
        label: str,
        *,
        as_xarray: Optional[bool] = False,
        as_qobj: Optional[bool] = False,
    ) -> Union[np.ndarray, Qobj]:
        eig_vals, eig_vecs = self.eig_states()
        ind = int(label)
        energy, state = eig_vals[ind], eig_vecs[ind]

        if as_qobj:
            q_dims = self.dim_hilbert #[qubit.dim_hilbert for qubit in self._qubits]

            qobj_state = Qobj(
                inpt=state,
                dims=[[q_dims], [1]],
                shape=[np.prod(q_dims), 1],
                type="ket",
            )

            return energy, qobj_state
        if as_xarray:
            state_arr = xr.DataArray(
                state,
                dims=["basis_ind"],
                coords=dict(label=label, energy=energy),
            )
            return state_arr
        return energy, state

    def mat_elements(
        self,
        operator: Union[str, np.ndarray],
        levels: Union[int, Iterable[int]] = None,
        *,
        expand: Optional[bool] = True,
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
                if callable(_op):
                    _op_params = _func_params(_op)
                    op = _op(expand=expand) if expand in _op_params else _op()
                else:
                    op = _op
                if not isinstance(op, np.ndarray):
                    raise ValueError("Obtained operator is not a numpy array")
                if len(op.shape) != 2:
                    raise ValueError("The operator must be a 2D array")

            else:
                raise ValueError(
                    "Given operator string not supported by"
                    + "the qubit or its basis {}".format(str(self.basis))
                )
        elif isinstance(operator, np.ndarray):
            op = operator
        else:
            raise ValueError("Incorrect operator provided")

        _, states = self.eig_states(levels=levels, expand=expand)

        mat_elems = get_mat_elem(op, states, states)

        if as_xarray:
            if levels is not None:
                if isinstance(levels, int):
                    levels_arr = list(range(levels))
                elif isinstance(levels, Iterable):
                    levels_arr = list(levels)
            else:
                levels_arr = list(range(self.dim_hilbert))

            data_arr = xr.DataArray(
                data=mat_elems,
                dims=["in_level", "out_level"],
                coords=dict(
                    in_level=levels_arr,
                    out_level=levels_arr,
                ),
                attrs=dict(
                    dim_hilbert=self.dim_hilbert,
                    basis=str(self.basis),
                    **self._qubit_attrs,
                ),
            )

            return data_arr
        return mat_elems

    def add_drive(self, drive: Union[Callable, "Drive"], **kwargs):
        if isinstance(drive, Callable):
            if "qubit" in kwargs:
                raise ValueError(
                    "Multiple values for the keyword arguement 'qubit' given"
                )
            _drive = drive(self, **kwargs)
            label = _drive.label
            if label in self._drives:
                raise ValueError("drive {} already in qubit drives".format(label))
            self._drives[label] = _drive
        elif isinstance(drive, Drive):
            if drive.label is not None:
                if drive.label in [q_drive.label for q_drive in self._drives]:
                    raise ValueError("Labeled drive already applied to this qubit")
                if drive.hilbert_dim != self.dim_hilbert:
                    raise ValueError(
                        "Dimensionality mismatch between the drive operator "
                        "(d={}) and the qubit (d={})".format(
                            drive.hilbert_dim, self.dim_hilbert
                        )
                    )
            label = drive.label
            if label in self._drives:
                raise ValueError("drive {} already in qubit drives".format(label))
            self._drives[label] = drive
        else:
            raise ValueError(
                "drive expected to be pysqkit.Drive or "
                "Callable, instead got {}".format(type(drive))
            )

    def couple_to(
        self,
        other_qubit: "Qubit",
        coupling: Optional[Union["Coupling", Callable]] = None,
        **kwargs,
    ) -> "QubitSystem":
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
                if "qubits" in kwargs:
                    raise ValueError(
                        "Multiple values for the keyword "
                        "arguement 'qubits' specified"
                    )
                return QubitSystem(qubits, coupling(qubits=qubits, **kwargs))
            elif isinstance(coupling, Coupling):
                if not all(q.label in coupling._qubits for q in qubits):
                    raise ValueError("Not all qubits defined in coupling term")
                return QubitSystem(qubits, coupling)
            else:
                raise ValueError(
                    "Coupling arguement expected to be pysqkit.CouplerTerm or "
                    "Callable, instead got {}".format(type(coupling))
                )
        else:
            return QubitSystem(qubits)

    def _parse_levels(self, levels: Union[int, Iterable[int]]):
        if levels is not None:
            if isinstance(levels, int):
                if levels < 1:
                    raise ValueError(
                        "Number of levels must be an integer greater than 1"
                    )
                if levels > self.dim_hilbert:
                    raise ValueError(
                        "Number of levels exceeds the "
                        "basis dimensionality of d={}".format(self.dim_hilbert)
                    )
                subset_inds = (0, levels - 1)
                sel_inds = None
            elif isinstance(levels, Iterable):
                subset_inds = min(levels), max(levels)
                if subset_inds[0] < 0:
                    raise ValueError(
                        "The lowest level index must be an integer "
                        "greater or equal to 0, "
                        "instead got {}".format(subset_inds[0])
                    )
                if subset_inds[1] >= self.dim_hilbert:
                    raise ValueError(
                        "The largest level index must be an integer "
                        "smaller then the basis dimensionality {}, "
                        "instead got {}".format(self.dim_hilbert, subset_inds[1])
                    )
                _inds = list(range(subset_inds[0], subset_inds[1] + 1))
                sel_inds = [_inds.index(level) for level in levels]
        else:
            subset_inds = None
            sel_inds = None

        return subset_inds, sel_inds

    def _qobj_oper(self, op, isherm=None, expand=True):
        if expand:
            dim = self.basis.sys_truncated_dims
        else:
            dim = [self.basis.truncated_dim]

        sys_dim = np.prod(dim)
        qobj = Qobj(
            inpt=op,
            dims=[dim, dim],
            shape=[sys_dim, sys_dim],
            type="oper",
            isherm=isherm,
        )
        return qobj



class Coupling:
    def __init__(
        self,
        prefactors: Union[float, complex, Iterable[Union[float, complex]]],
        operators: Iterable[Dict[str, np.ndarray]],
        qubits: Optional[List[str]] = None,
    ):

        if isinstance(operators, dict):
            self._ops = [operators]
        elif isinstance(operators, Iterable):
            self._ops = list(operators)

        for op in self._ops:
            for key, val in op.items():
                if not isinstance(key, str):
                    raise ValueError("Operator keys must be qubit labels of type str")
                if not isinstance(val, np.ndarray):
                    raise ValueError("The operators must be np.ndarray type")

        if qubits is None:
            qubit_set = set(chain.from_iterable(op.keys() for op in self._ops))
            self._qubits = sorted(list(qubit_set))
        else:
            involved_qubits = set(chain.from_iterable(op.keys() for op in self._ops))

            qubit_set = set(qubits)
            if len(qubit_set) != len(qubits):
                raise ValueError("There are duplicate labels in the qubits")
            if qubit_set != involved_qubits:
                raise ValueError(
                    "Qubit labels have a mismatch with the operator labels"
                )
            self._qubits = qubits

        self._hilbert_dims = {}
        for op in self._ops:
            for qubit, qubit_op in op.items():
                if len(qubit_op.shape) != 2 or qubit_op.shape[0] != qubit_op.shape[1]:
                    raise ValueError("Each operator must be a square matrix")

                if qubit in self._hilbert_dims:
                    if self._hilbert_dims[qubit] != qubit_op.shape[0]:
                        raise ValueError(
                            "Mismatch in the dimensionality of the qubit operators in different terms"
                        )
                else:
                    self._hilbert_dims[qubit] = qubit_op.shape[0]

        if isinstance(prefactors, (float, int, complex)):
            self._prefactors = [prefactors] * len(self._ops)
        elif isinstance(prefactors, Iterable):
            if len(prefactors) != len(self._ops):
                raise ValueError(
                    "Number of provided operators does not correspond to the number of terms in the Hamiltonian"
                )
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
        hamiltonian = np.zeros((dim, dim), dtype="complex")

        for prefactor, op in self.hamiltonian_terms():
            hamiltonian += prefactor * op

        if np.all(np.real(hamiltonian)):
            return hamiltonian.real
        return hamiltonian

    def hamiltonian(self) -> np.ndarray:
        hamil = self._get_hamiltonian()
        return hamil

    def hamiltonian_terms(
        self, *, tensor_ops=True
    ) -> Iterable[
        Tuple[Union[float, complex], Union[np.ndarray, Dict[str, np.ndarray]]]
    ]:
        if tensor_ops:
            for prefactor, term_ops in zip(self._prefactors, self._ops):
                inv_qs = list(term_ops.keys())
                total_ops = list(
                    term_ops[q] if q in inv_qs else np.eye(self.hilbert_dims[q])
                    for q in self._qubits
                )
                yield prefactor, tensor_prod(total_ops)
        else:
            for prefactor, term_ops in zip(self._prefactors, self._ops):
                yield prefactor, term_ops


class Drive:
    def __init__(
        self,
        label: str,
        operator: np.ndarray,
        pulse: Union[np.ndarray, Callable],
        pulse_shape: Optional[Callable] = None,
        *,
        hilbert_dim: Optional[Union[int, Tuple[int]]] = None,
    ):
        if not isinstance(label, str):
            raise ValueError(
                "drive_label expected to be str, " "instead got {}".format(type(label))
            )
        self._label = label

        if not isinstance(operator, np.ndarray):
            raise ValueError("The operators must be np.ndarray type")
        if len(operator.shape) != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError("operator must be a square matrix")
        self._op = operator

        op_dim = operator.shape[0]
        if hilbert_dim is not None:
            if np.product(hilbert_dim) != op_dim:
                raise ValueError(
                    "Mismatch between provided hilbert_dim={} and "
                    "dimensionality of operator: {}".format(hilbert_dim, op_dim)
                )
            self._hilbert_dim = hilbert_dim
        else:
            self._hilbert_dim = op_dim

        self._params = {}

        if pulse_shape is not None:
            if not isinstance(pulse_shape, Callable):
                raise ValueError(
                    "pulse_shape expected to be callable, "
                    "instead got {}".format(type(pulse))
                )
            shape_params = _func_params(pulse_shape)

            self._shape_params = set(shape_params.keys())
            self._params.update(shape_params)
        self._pulse_shape = pulse_shape

        if isinstance(pulse, Callable):
            pulse_params = _func_params(pulse)

            self._pulse_params = set(pulse_params.keys())
            self._params.update(pulse_params)
        elif not isinstance(pulse, np.ndarray):
            raise ValueError(
                "pulse expected to be np.ndarray or callable, "
                "instead got {}".format(type(pulse))
            )
        self._pulse = pulse

    def __copy__(self) -> "Drive":
        drive_copy = self.__class__(
            self._label,
            self._op,
            self._pulse,
            self._pulse_shape,
            hilbert_dim=self._hilbert_dim,
        )
        drive_copy._params = self._params
        return drive_copy

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new_label: str) -> None:
        if not isinstance(new_label, str):
            raise ValueError(
                "drive_label expected to be str, "
                "instead got {}".format(type(new_label))
            )
        self._label = new_label

    @property
    def hilbert_dim(self):
        return self._hilbert_dim

    @property
    def params(self):
        return self._params

    @property
    def free_params(self):
        free_params = [par for par, val in self._params.items() if val is None]
        return free_params

    def set_params(self, **params):
        for param, val in params.items():
            if param not in self._params:
                raise ValueError("parameter {} not in drive parameters".format(param))
            self._params[param] = val

    def _get_hamiltonian(self, include_pulse=False, **params) -> np.ndarray:
        hamil = self._op
        if include_pulse:
            pulse = self.eval_pulse(**params)
            if isinstance(pulse, (float, complex)):
                return pulse * hamil
            elif isinstance(pulse, np.ndarray):
                if len(pulse.shape) != 1:
                    raise ValueError(
                        "Evaluated pulse expected to be vector, "
                        "instead got {}-dimensional array".format(len(pulse.shape))
                    )
                return np.einsum("i, jk -> ijk", pulse, hamil)
            else:
                raise ValueError("Unsupported pulse type {}".format(type(pulse)))
        return hamil

    def hamiltonian(
        self,
        *,
        as_qobj=False,
        include_pulse=False,
        **params,
    ) -> np.ndarray:
        hamil = self._get_hamiltonian(include_pulse, **params)
        if as_qobj:
            dim = self._hilbert_dim
            hamil_dims = len(hamil.shape)
            if hamil_dims == 2:
                qobj_op = Qobj(
                    inpt=hamil,
                    dims=[dim, dim],
                    shape=hamil.shape,
                    type="oper",
                    isherm=True,
                )
                return qobj_op
            elif hamil_dims == 3:
                qobj_ops = []
                for hamil_op in hamil:
                    qobj_op = Qobj(
                        inpt=hamil_op,
                        dims=[dim, dim],
                        shape=hamil.shape,
                        type="oper",
                        isherm=True,
                    )
                    qobj_ops.append(qobj_op)
                return qobj_ops
            else:
                raise ValueError(
                    "Unsupported hamiltonian dimensionality for qobj conversion"
                )
        return hamil

    def eval_pulse(self, **params):
        if isinstance(self._pulse, np.ndarray):
            return self._pulse

        free_params = set(self.free_params)
        given_params = set(params)

        unset_params = [par for par in free_params if par not in given_params]
        if unset_params:
            raise ValueError("Drive parameters not specified: {}".format(unset_params))

        set_params = {**self.params, **params}
        if self._pulse_shape is not None:
            pulse_params = {
                param: val
                for param, val in set_params.items()
                if param in self._pulse_params
            }
            pulse = self._pulse(**pulse_params)

            shape_params = {
                param: val
                for param, val in set_params.items()
                if param in self._shape_params
            }
            pulse_shape = self._pulse_shape(**shape_params)
            return np.multiply(pulse_shape, pulse)
        return self._pulse(**set_params)


class QubitSystem:
    def __init__(
        self,
        qubits: Iterable[Qubit],
        coupling: Optional[Union[Coupling, Iterable[Coupling]]] = None,
    ) -> None:
        qubit_labels = []

        for qubit in qubits:
            if not isinstance(qubit, Qubit):
                raise ValueError(
                    "Each qubit must be a pysqkit.Qubit object, "
                    "instead got {}".format(type(qubit))
                )

            if qubit.label in qubit_labels:
                raise ValueError(
                    "Multiple qubits share the same label, "
                    "please ensure each qubit is uniquely labeled."
                )

            if qubit.basis.is_subbasis:
                raise ValueError(
                    "The basis of qubit {} is a subbasis, please ensure qubits"
                    "are not part of a system already".format(qubit.label)
                )

            qubit_labels.append(qubit.label)

        self._qubits = [copy(qubit) for qubit in qubits]
        self._labels = qubit_labels

        sys_dims = [qubit.basis.truncated_dim for qubit in self._qubits]
        for qubit_ind, qubit in enumerate(self._qubits):
            qubit.basis.embed(qubit_ind, sys_dims)
            if qubit.is_driven:
                for qubit_drive in qubit.drives.values():
                    expanded_op = qubit.basis.expand_op(qubit_drive._op)
                    qubit_drive._op = expanded_op
                    qubit_drive._hilbert_dim = sys_dims

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
                            "{}, which is not part of the system".format(q)
                        )
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
                raise KeyError("Qubit {} is not part of the state".format(q_label))
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
                "Can't check if a {} object is in system".format(type(qubit))
            )

    def __iter__(self) -> Iterable[Qubit]:
        yield from self._qubits

    @property
    def size(self):
        return len(self._qubits)

    @property
    def qubits(self) -> List[Qubit]:
        return self._qubits

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def dim_hilbert(self) -> int:
        return np.prod([q.dim_hilbert for q in self.qubits])

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

    def bare_hamiltonian(self, *, as_qobj=False) -> np.ndarray:
        bare_hamiltonians = [qubit.hamiltonian() for qubit in self._qubits]
        bare_hamil = np.sum(bare_hamiltonians, axis=0)

        if as_qobj:
            sys_dims = [qubit.basis.truncated_dim for qubit in self._qubits]

            qobj_op = Qobj(
                inpt=bare_hamil,
                dims=[sys_dims, sys_dims],
                shape=bare_hamil.shape,
                type="oper",
                isherm=True,
            )
            return qobj_op
        return bare_hamil

    def int_hamiltonian(self, *, as_qobj=False) -> np.ndarray:
        sys_dim = np.prod([qubit.basis.truncated_dim for qubit in self._qubits])

        int_hamiltonian = np.zeros((sys_dim, sys_dim), dtype=complex)

        if self._coupling:
            for coup in self._coupling:
                coup_terms = coup.hamiltonian_terms(tensor_ops=False)
                for prefactor, term_ops in coup_terms:
                    ops = []
                    for qubit, qubit_op in term_ops.items():
                        q_ind = self.index(qubit)
                        op = self._qubits[q_ind].basis.expand_op(qubit_op)
                        ops.append(op)

                    int_hamiltonian += prefactor * reduce(np.matmul, ops)

        if as_qobj:
            sys_dims = [qubit.basis.truncated_dim for qubit in self._qubits]

            qobj_op = Qobj(
                inpt=int_hamiltonian,
                dims=[sys_dims, sys_dims],
                shape=int_hamiltonian.shape,
                type="oper",
                isherm=True,
            )
            return qobj_op

        return int_hamiltonian

    def hamiltonian(self, *, as_qobj=False) -> np.ndarray:
        bare_hamiltonian = self.bare_hamiltonian(as_qobj=as_qobj)
        int_hamiltonian = self.int_hamiltonian(as_qobj=as_qobj)

        return bare_hamiltonian + int_hamiltonian

    def _get_eig_vals(
        self, subset_inds: Tuple[int], bare_system: Optional[bool] = False
    ) -> np.ndarray:
        hamil = self.bare_hamiltonian() if bare_system else self.hamiltonian()
        eig_vals = la.eigh(hamil, eigvals_only=True, subset_by_index=subset_inds)
        return eig_vals

    def _get_eig_states(
        self, subset_inds: Tuple[int], bare_system: Optional[bool] = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        hamil = self.bare_hamiltonian() if bare_system else self.hamiltonian()
        eig_vals, eig_vecs = la.eigh(
            hamil, eigvals_only=False, subset_by_index=subset_inds
        )
        return eig_vals, eig_vecs.T

    def eig_energies(
        self,
        levels: Optional[Union[int, Iterable[int]]] = None,
        bare_system: Optional[bool] = False,
    ) -> np.ndarray:
        subset_inds, sel_inds = self._parse_levels(levels)
        eig_vals = self._get_eig_vals(subset_inds, bare_system)
        if sel_inds:
            return order_vecs(eig_vals[sel_inds])
        return order_vecs(eig_vals)

    def eig_states(
        self,
        levels: Optional[Union[int, Iterable[int]]] = None,
        bare_system: Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        subset_inds, sel_inds = self._parse_levels(levels)

        eig_vals, eig_vecs = self._get_eig_states(subset_inds, bare_system)
        if sel_inds:
            return order_vecs(eig_vals[sel_inds], eig_vecs[sel_inds])
        return order_vecs(eig_vals, eig_vecs)

    def state_index(
        self,
        label: str,
    ) -> int:

        if len(label) != self.size:
            raise ValueError(
                "label describe a {}-qubit state, while system "
                "contains only {} qubits".format(len(label), self.size)
            )
        levels = [int(level) for level in label]

        bare_energy = 0
        for level, qubit in zip(levels, self._qubits):
            if int(level) >= qubit.dim_hilbert:
                raise ValueError(
                    "label specifies level {} for qubit {}, "
                    "but qubit is {}-dimensional".format(
                        level, qubit.label, qubit.dim_hilbert
                    )
                )
            bare_energy += qubit.eig_energies(levels=[level], expand=False)

        bare_energies = self.eig_energies(bare_system=True)

        ind = np.argmin(np.abs(bare_energies - bare_energy))
        return ind
    

    def state(
        self,
        label: str,
        *,
        as_xarray: Optional[bool] = False,
        as_qobj: Optional[bool] = False,
    ) -> Union[np.ndarray, Qobj]:
        eig_vals, eig_vecs = self.eig_states()
        ind = self.state_index(label)
        energy, state = eig_vals[ind], eig_vecs[ind]

        if as_qobj:
            q_dims = [qubit.dim_hilbert for qubit in self._qubits]

            qobj_state = Qobj(
                inpt=state,
                dims=[q_dims, [1] * self.size],
                shape=[np.prod(q_dims), 1],
                type="ket",
            )

            return energy, qobj_state
        if as_xarray:
            state_arr = xr.DataArray(
                state,
                dims=["basis_ind"],
                coords=dict(label=label, energy=energy),
            )
            return state_arr
        return energy, state
    
    def all_state_labels(self) -> List:

        """
        Description
        ------------------------------------------------------------------
        Returns the state labels as a list of strings. Example
        ['00', '01', '02', ...]
        """

        dims_list = [qubit.dim_hilbert for qubit in self._qubits]
        lev_tmp = []
        for dim in dims_list:
            lev_tmp.append([x for x in range(dim)])
        states_tmp = [element for element in product(*lev_tmp)]
        states = ["".join(map(str, states_tmp[x])) \
            for x in range(len(states_tmp))]
        return states
    
    def states_as_dict(
        self, 
        as_qobj: Optional[bool] = False
        ) -> dict:

        """
        Description
        ----------------------------------------------------------------------
        Returns the eigenenergies and eigenstates as a dictionary where 
        the keys are the levels indices.
        """

        states = self.all_state_labels()
        energy_dict = {}
        eig_vals, eig_vecs = self.eig_states()
        for state_label in states:
            ind = self.state_index(state_label)
            energy, state = eig_vals[ind], eig_vecs[ind]
            if as_qobj:
                q_dims = [qubit.dim_hilbert for qubit in self._qubits]

                qobj_state = Qobj(
                    inpt=state,
                    dims=[q_dims, [1] * self.size],
                    shape=[np.prod(q_dims), 1],
                    type="ket",
                )
                energy_dict[state_label] = (energy, qobj_state)
            else:
                energy_dict[state_label] = (energy, state)
        
        return energy_dict

    def diagonalizing_unitary(
        self,
        as_qobj: Optional[bool] = False
    ) -> Union[np.ndarray, Qobj]:
        
        """
        Description
        ----------------------------------------------------------------------
        Returns the unitary that diagonalizes the coupled Hamiltonian
        """
        u = np.zeros([self.dim_hilbert, self.dim_hilbert], dtype=complex)

        states_dict = self.states_as_dict()

        col = 0

        for key in states_dict.keys():
            u[:, col] = states_dict[key][1]
            col += 1
        
        if as_qobj:
            sys_dims = [qubit.basis.truncated_dim for qubit in self._qubits]
            qobj_op = Qobj(
                inpt=u,
                dims=[sys_dims, sys_dims],
                shape=u.shape,
                type="oper",
                isherm=True,
            )
            return qobj_op
        else:
            return u

 


    def mat_elements(
        self,
        operator: Union[str, np.ndarray],
        levels: Union[int, Iterable[int]] = None,
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
                    "Given operator string not supported by"
                    + "the qubit or its basis {}".format(str(self.basis))
                )
        elif isinstance(operator, np.ndarray):
            op = operator
        else:
            raise ValueError("Incorrect operator provided")

        _, states = self.eig_states(levels=levels)

        mat_elems = get_mat_elem(op, states, states)

        if levels is not None:
            if isinstance(levels, int):
                levels_arr = list(range(levels))
            elif isinstance(levels, Iterable):
                levels_arr = list(levels)
        else:
            levels_arr = list(range(self.dim_hilbert))

        if as_xarray:
            data_arr = xr.DataArray(
                data=mat_elems,
                dims=["in_level", "out_level"],
                coords=dict(
                    in_level=levels_arr,
                    out_level=levels_arr,
                ),
                attrs=dict(
                    dim_hilbert=self.dim_hilbert,
                ),
            )

            return data_arr
        return mat_elems

    def _parse_levels(self, levels: Union[int, Iterable[int]]):
        if levels is not None:
            if isinstance(levels, int):
                if levels < 1:
                    raise ValueError(
                        "Number of levels must be an integer greater than 1"
                    )
                if levels > self.dim_hilbert:
                    raise ValueError(
                        "Number of levels exceeds the "
                        "basis dimensionality of d={}".format(self.dim_hilbert)
                    )
                subset_inds = (0, levels - 1)
                sel_inds = None
            elif isinstance(levels, Iterable):
                subset_inds = min(levels), max(levels)
                if subset_inds[0] < 0:
                    raise ValueError(
                        "The lowest level index must be an integer "
                        "greater or equal to 0, "
                        "instead got {}".format(subset_inds[0])
                    )
                if subset_inds[1] >= self.dim_hilbert:
                    raise ValueError(
                        "The largest level index must be an integer "
                        "smaller then the basis dimensionality {}, "
                        "instead got {}".format(self.dim_hilbert, subset_inds[1])
                    )
                _inds = list(range(subset_inds[0], subset_inds[1] + 1))
                sel_inds = [_inds.index(level) for level in levels]
        else:
            subset_inds = None
            sel_inds = None

        return subset_inds, sel_inds


def _func_params(func):
    sig = signature(func)
    params = {}

    for param in sig.parameters.values():
        if param.default is param.empty:
            params[param.name] = None
        else:
            params[param.name] = param.default
    return params
