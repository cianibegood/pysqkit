from typing import Union, Iterable

from ..systems import Qubit, Coupling


def capacitive_coupling(
        qubits: Iterable[Qubit],
        strength: Union[float, complex],
) -> Coupling:
    if len(qubits) != 2 or any(not isinstance(q, Qubit) for q in qubits):
        raise ValueError("Only pair of qubits can be coupled")
    q_labels = [q.label for q in qubits]

    if None in q_labels:
        raise ValueError("Qubits must be labeled")

    if q_labels[0] == q_labels[1]:
        raise ValueError("Qubits must have distinct labels")

    operators = {q.label: q.charge_op() for q in qubits}

    coupling = Coupling(
        prefactors=strength,
        operators=operators,
        qubits=q_labels
    )

    return coupling
