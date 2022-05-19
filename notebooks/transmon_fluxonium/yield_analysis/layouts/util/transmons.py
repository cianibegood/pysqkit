from optparse import Option
from typing import List, Optional
import numpy as np

from ..layout import Layout


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

        tar_freq = group_freqs[freq_group - 1]
        layout.set_param("target_freq", qubit, tar_freq)

        anharm = group_anharms[freq_group - 1]
        layout.set_param("anharm", qubit, anharm)


def sample_freqs(
    layout: Layout, seed: int, freq_var: float, *, relative_var: Optional[bool] = False
):
    rng = np.random.default_rng(seed)

    qubits = layout.get_qubits()
    for qubit in qubits:
        tar_freq = layout.param("target_freq", qubit)
        if relative_var:
            freq = rng.normal(tar_freq, freq_var * tar_freq)
        else:
            freq = rng.normal(tar_freq, freq_var)
        layout.set_param("freq", qubit, freq)


def get_num_collisions(layout: Layout, bounds: List[float]) -> List[int]:
    num_collisions = np.zeros(7, dtype=int)

    anc_qubits = layout.get_qubits(role="anc")
    for anc_qubit in anc_qubits:
        anc_freq = layout.param("freq", anc_qubit)
        anc_anharm = layout.param("anharm", anc_qubit)

        data_qubits = layout.get_neighbors(anc_qubit)

        for data_qubit in data_qubits:
            data_freq = layout.param("freq", data_qubit)
            data_anharm = layout.param("anharm", data_qubit)

            if abs(anc_freq - data_freq) < bounds[0]:
                num_collisions[0] += 1

            if abs(anc_freq - data_freq - data_anharm) < bounds[1]:
                num_collisions[1] += 1

            if abs(data_freq - anc_freq - anc_anharm) < bounds[1]:
                num_collisions[1] += 1

            if layout.param("target_freq", anc_qubit) > layout.param(
                "target_freq", data_qubit
            ):
                ctrl_qubit, tar_qubit = anc_qubit, data_qubit
                ctrl_freq, tar_freq = anc_freq, data_freq
                ctrl_anharm = anc_anharm

            else:
                ctrl_qubit, tar_qubit = data_qubit, anc_qubit
                ctrl_freq, tar_freq = data_freq, anc_freq
                ctrl_anharm = data_anharm

            if ctrl_freq < tar_freq:
                num_collisions[2] += 1
            if tar_freq < (ctrl_freq + ctrl_anharm):
                num_collisions[2] += 1

            ctrl_02_freq = 2 * ctrl_freq + ctrl_anharm
            if abs(2 * tar_freq - ctrl_02_freq) < bounds[2]:
                num_collisions[3] += 1

            spectator_qubits = layout.get_neighbors(ctrl_qubit)
            for spec_qubit in spectator_qubits:
                if spec_qubit != tar_qubit:
                    spec_freq = layout.param("freq", spec_qubit)
                    spec_anharm = layout.param("anharm", spec_qubit)

                    if abs(tar_freq - spec_freq) < bounds[3]:
                        num_collisions[4] += 1

                    spec_12_freq = spec_freq + spec_anharm
                    if abs(tar_freq - spec_12_freq) < bounds[4]:
                        num_collisions[5] += 1

                    if abs(tar_freq + spec_freq - ctrl_02_freq) < bounds[5]:
                        num_collisions[6] += 1

    return num_collisions


def any_collisions(layout: Layout, bounds: List[float]) -> List[int]:
    if len(bounds) != 6:
        raise ValueError("Expected only 6 bounds to be provided.")

    anc_qubits = layout.get_qubits(role="anc")
    for anc_qubit in anc_qubits:
        anc_freq = layout.param("freq", anc_qubit)
        anc_anharm = layout.param("anharm", anc_qubit)

        data_qubits = layout.get_neighbors(anc_qubit)

        for data_qubit in data_qubits:
            data_freq = layout.param("freq", data_qubit)
            data_anharm = layout.param("anharm", data_qubit)

            if abs(anc_freq - data_freq) < bounds[0]:
                return True

            if abs(anc_freq - data_freq - data_anharm) < bounds[1]:
                return True

            if abs(data_freq - anc_freq - anc_anharm) < bounds[1]:
                return True

            if layout.param("target_freq", anc_qubit) > layout.param(
                "target_freq", data_qubit
            ):
                ctrl_qubit, tar_qubit = anc_qubit, data_qubit
                ctrl_freq, tar_freq = anc_freq, data_freq
                ctrl_anharm = anc_anharm

            else:
                ctrl_qubit, tar_qubit = data_qubit, anc_qubit
                ctrl_freq, tar_freq = data_freq, anc_freq
                ctrl_anharm = data_anharm

            if ctrl_freq < tar_freq:
                return True
            if tar_freq < (ctrl_freq + ctrl_anharm):
                return True

            ctrl_02_freq = 2 * ctrl_freq + ctrl_anharm
            if abs(2 * tar_freq - ctrl_02_freq) < bounds[2]:
                return True

            spectator_qubits = layout.get_neighbors(ctrl_qubit)
            for spec_qubit in spectator_qubits:
                if spec_qubit != tar_qubit:
                    spec_freq = layout.param("freq", spec_qubit)
                    spec_anharm = layout.param("anharm", spec_qubit)

                    if abs(tar_freq - spec_freq) < bounds[3]:
                        return True

                    spec_12_freq = spec_freq + spec_anharm
                    if abs(tar_freq - spec_12_freq) < bounds[4]:
                        return True

                    if abs(tar_freq + spec_freq - ctrl_02_freq) < bounds[5]:
                        return True
    return False
