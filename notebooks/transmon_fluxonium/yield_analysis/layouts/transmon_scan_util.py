from typing import List
import numpy as np

from .layout import Layout
from .transmon_collisions import (
    address_collision,
    cross_res_collision,
    spectator_collision,
)


def sample_freqs(layout: Layout, seed: int, freq_var: float):
    rng = np.random.default_rng(seed)

    qubits = layout.get_qubits()
    for qubit in qubits:
        target_freq = layout.param("target_freq", qubit)
        freq = rng.normal(target_freq, freq_var * target_freq)
        layout.set_param("freq", qubit, freq)


def get_collisions(layout: Layout) -> List[int]:
    collision_counter = [0, 0, 0]

    anc_qubits = layout.get_qubits(role="anc")
    for anc_qubit in anc_qubits:
        anc_freq = layout.param("freq", anc_qubit)
        anc_anharm = layout.param("anharm", anc_qubit)

        data_qubits = layout.get_neighbors(anc_qubit)

        for data_qubit in data_qubits:
            data_freq = layout.param("freq", data_qubit)
            data_anharm = layout.param("anharm", data_qubit)

            if address_collision(anc_freq, anc_anharm, data_freq, data_anharm):
                collision_counter[0] += 1

            if layout.param("target_freq", anc_qubit) > layout.param(
                "target_freq", data_qubit
            ):
                control_qubit, target_qubit = anc_qubit, data_qubit
                control_freq, target_freq = anc_freq, data_freq
                control_anharm = anc_anharm

            else:
                control_qubit, target_qubit = data_qubit, anc_qubit
                control_freq, target_freq = data_freq, anc_freq
                control_anharm = data_anharm

            if cross_res_collision(control_freq, control_anharm, target_freq):
                collision_counter[1] += 1

            spectators = layout.get_neighbors(control_qubit)
            for spectator in spectators:
                if spectator != target_qubit:
                    spectator_freq = layout.param("freq", spectator)
                    spectator_anharm = layout.param("anharm", spectator)

                    if spectator_collision(
                        control_freq,
                        control_anharm,
                        target_freq,
                        spectator_freq,
                        spectator_anharm,
                    ):
                        collision_counter[2] += 1
    return collision_counter
