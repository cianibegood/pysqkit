from typing import Union

from ..qubits import Qubit
from .coupler import CouplingTerm


def capacitive_coupling(
        strength: Union[float, complex],
        qubit_0: Qubit,
        qubit_1: Qubit
) -> CouplingTerm:

    q_labels = [qubit_0.label, qubit_1.label]

    if None in q_labels:
        raise ValueError("Qubits must be labeled")

    if q_labels[0] == q_labels[1]:
        raise ValueError("Qubits must have distinct labels")

    operators = {q.label: q.basis.charge_op for q in (qubit_0, qubit_1)}

    coupling = CouplingTerm(
        prefactors=strength,
        operators=operators,
        qubits=q_labels
    )

    return coupling
