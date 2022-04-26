from typing import List
import numpy as np

from .layout import Layout
from .transmon_collisions import (
    address_collision,
    cross_res_collision,
    spectator_collision,
)


def set_freq_groups(layout: Layout) -> None:
    num_groups = 5
    group_assign_order = ["north_east", "north_west", "south_east", "south_west"]

    qubits = layout.get_qubits()
    init_qubit = qubits.pop()
    init_group = 0

    assigned_qubits = set()

    def dfs_assign(qubit, freq_group):
        if qubit not in assigned_qubits:
            layout.set_param("freq_group", qubit, freq_group + 1)
            assigned_qubits.add(qubit)

            neighbors_dict = layout.param("neighbors", qubit)
            for direction, neighbour in neighbors_dict.items():
                if (
                    neighbour
                ):  # Supports more explicit syntax where absent neighbours are set as none
                    group_shift = group_assign_order.index(direction) + 1
                    neighbor_freq_group = (freq_group + group_shift) % num_groups
                    dfs_assign(neighbour, neighbor_freq_group)

    dfs_assign(init_qubit, init_group)


def set_target_freqs(
    layout: Layout, group_freqs: List[float], group_anharms: List[float]
) -> None:
    if len(group_freqs) != 5:
        raise ValueError(
            "5 distinct qubit frequencies are required for the square layout."
        )
    qubits = layout.get_qubits()

    for qubit in qubits:
        try:
            freq_group = layout.param("freq_group", qubit)
        except KeyError:
            raise ValueError(
                f"Layout does not define a frequency group for qubit {qubit}."
            )

        if freq_group is None:
            raise ValueError(
                f"Layout does not define a frequency group for qubit {qubit}."
            )

        target_freq = group_freqs[freq_group - 1]
        layout.set_param("target_freq", qubit, target_freq)

        anharm = group_anharms[freq_group - 1]
        layout.set_param("anharm", qubit, anharm)


def sample_freqs(layout: Layout, seed: int, freq_var: float):
    rng = np.random.default_rng(seed)

    qubits = layout.get_qubits()
    for qubit in qubits:
        target_freq = layout.param("target_freq", qubit)
        freq = rng.normal(target_freq, freq_var * target_freq)
        layout.set_param("freq", qubit, freq)


def get_collisions(layout: Layout) -> List[int]:
    num_address_collisions = 0
    num_cross_res_collisions = 0
    num_spectator_collisions = 0

    anc_qubits = layout.get_qubits(role="anc")
    for anc_qubit in anc_qubits:
        anc_freq = layout.param("freq", anc_qubit)
        anc_anharm = layout.param("anharm", anc_qubit)

        data_qubits = layout.get_neighbors(anc_qubit)

        for data_qubit in data_qubits:
            data_freq = layout.param("freq", data_qubit)
            data_anharm = layout.param("anharm", data_qubit)

            if address_collision(anc_freq, anc_anharm, data_freq, data_anharm):
                num_address_collisions += 1

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
                num_cross_res_collisions += 1

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
                        num_spectator_collisions += 1
    result = np.array(
        [num_address_collisions, num_cross_res_collisions, num_spectator_collisions],
        dtype=int,
    )
    return result
