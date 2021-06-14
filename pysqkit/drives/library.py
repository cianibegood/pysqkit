from ..systems import Qubit


def microwave_drive(
        qubit: Qubit,
        amplitude: float,
        frequency: float,
        phase: float
):
    if not isinstance(qubit, Qubit):
        raise ValueError(
            "qubit expected to a Qubit instance, "
            "instead got {}".format(type(qubit))
            )

    if qubit.label is None:
        raise ValueError("Qubit must be labeled")

    coupling = None

    return coupling
