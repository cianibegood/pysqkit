from typing import Dict, List
import numpy as np

from ..layout import Layout


def _combine_dirs(direction_i, direction_j):
    v_dir_i, h_dir_i = direction_i.split("_")
    v_dir_j, h_dir_j = direction_j.split("_")

    if v_dir_i == v_dir_j:
        if h_dir_i != h_dir_j:
            return "vertical"
    else:
        if h_dir_i == h_dir_j:
            return "horizontal"
    return "diagonal"


def set_freq_groups(layout: Layout) -> None:
    transmons = layout.get_qubits(qubit_type="transmon")
    init_transmon = transmons.pop()
    init_group = 0

    assigned_transmons = set()

    def dfs_assign(transmon, freq_group):
        if transmon not in assigned_transmons:
            layout.set_param("freq_group", transmon, freq_group + 1)
            assigned_transmons.add(transmon)

            neighbours = layout.param("neighbors", transmon)
            for dir, neighbor in neighbours.items():
                if neighbor:
                    next_neigbours = layout.param("neighbors", neighbor)
                    for next_dir, next_neighbor in next_neigbours.items():
                        if next_neighbor and next_neighbor != transmon:
                            combined_dir = _combine_dirs(dir, next_dir)
                            if combined_dir == "horizontal":
                                next_neighbor_group = (freq_group + 2) % 4
                                dfs_assign(next_neighbor, next_neighbor_group)
                            elif combined_dir == "vertical":
                                offset = 2 * (freq_group // 2)
                                shifted_group = ((freq_group % 2) + 1) % 2
                                next_neighbor_group = shifted_group + offset
                                dfs_assign(next_neighbor, next_neighbor_group)

    dfs_assign(init_transmon, init_group)

    fluxoniums = layout.get_qubits(qubit_type="fluxonium")
    for fluxonium in fluxoniums:
        layout.set_param("freq_group", fluxonium, 0)


def set_transmon_target_freqs(
    layout: Layout, group_freqs: List[float], group_anharms: List[float]
) -> None:
    if len(group_freqs) != 4:
        raise ValueError(
            "5 distinct qubit frequencies are required for the square layout."
        )
    transmons = layout.get_qubits(qubit_type="transmon")

    for transmon in transmons:
        try:
            freq_group = layout.param("freq_group", transmon)
        except KeyError:
            raise ValueError(
                f"Layout does not define a frequency group for qubit {transmon}."
            )

        if freq_group is None:
            raise ValueError(
                f"Layout does not define a frequency group for qubit {transmon}."
            )

        if freq_group == 0:
            raise ValueError("Only fluxonia can have frequency group 0")

        target_freq = group_freqs[freq_group - 1]
        layout.set_param("target_freq", transmon, target_freq)

        anharm = group_anharms[freq_group - 1]
        layout.set_param("anharm", transmon, anharm)


def set_fluxonium_target_params(layout: Layout, trans_freqs: Dict[str, float],) -> None:

    collision_freqs = ["freq_12", "freq_03", "freq_04", "freq_05", "freq_15"]
    for collision_freq in collision_freqs:
        if collision_freqs not in trans_freqs:
            raise ValueError(
                f"trans_freqs does not contain collision frequency {collision_freq}"
            )

    fluxoniums = layout.get_qubits(qubit_type="fluxonium")

    for fluxonium in fluxoniums:
        for collision_freq in collision_freqs:
            layout.set_param(collision_freq, fluxonium, trans_freqs[collision_freq])


def sample_params(
    layout: Layout, seed: int, resist_var: float,
):
    rng = np.random.default_rng(seed)

    freq_var = 0.5 * resist_var

    transmons = layout.get_qubits(qubit_type="transmon")
    for transmon in transmons:
        target_freq = layout.param("target_freq", transmon)
        freq = rng.normal(target_freq, freq_var * target_freq)
        layout.set_param("freq", transmon, freq)


def get_num_collisions(layout: Layout, bounds: List[float]) -> List[int]:
    num_collisions = np.zeros(7, dtype=int)

    fluxoniums = layout.get_qubits(qubit_type="fluxonium")
    for fluxonium in fluxoniums:
        ctrl_freq_12 = layout.param("freq_12", fluxonium)
        ctrl_freq_03 = layout.param("freq_03", fluxonium)
        ctrl_freq_04 = layout.param("freq_04", fluxonium)
        ctrl_freq_05 = layout.param("freq_05", fluxonium)
        ctrl_freq_15 = layout.param("freq_15", fluxonium)

        transmons = layout.get_neighbors(fluxonium)
        for tar_tmon in transmons:
            tar_freq = layout.param("freq", tar_tmon)

            if abs(tar_freq - ctrl_freq_12) < bounds[0]:
                num_collisions[0] += 1

            if abs(tar_freq - ctrl_freq_03) < bounds[0]:
                num_collisions[0] += 1

            if tar_freq < ctrl_freq_12:
                num_collisions[1] += 1

            if tar_freq > ctrl_freq_03:
                num_collisions[1] += 1

            if abs(2 * tar_freq - ctrl_freq_04) < bounds[1]:
                num_collisions[2] += 1

            if abs(3 * tar_freq - ctrl_freq_05) < bounds[2]:
                num_collisions[3] += 1

            for spec_tmon in transmons:
                if spec_tmon != tar_tmon:
                    spec_freq = layout.param("freq", spec_tmon)
                    spec_anharm = layout.param("anharm", spec_tmon)

                    if abs(tar_freq - spec_freq) < bounds[3]:
                        num_collisions[4] += 1

                    spec_12_freq = spec_freq + spec_anharm
                    if abs(tar_freq - spec_12_freq) < bounds[4]:
                        num_collisions[5] += 1

                    if abs(tar_freq + spec_freq - ctrl_freq_04) < bounds[5]:
                        num_collisions[6] += 1

    return num_collisions


def any_collisions(layout: Layout, bounds: List[float]) -> List[int]:
    if len(bounds) != 6:
        raise ValueError("Expected only 6 bounds to be provided.")

    fluxoniums = layout.get_qubits(qubit_type="fluxonium")
    for fluxonium in fluxoniums:
        ctrl_freq_12 = layout.param("freq_12", fluxonium)
        ctrl_freq_03 = layout.param("freq_03", fluxonium)
        ctrl_freq_04 = layout.param("freq_04", fluxonium)
        ctrl_freq_05 = layout.param("freq_05", fluxonium)
        ctrl_freq_15 = layout.param("freq_15", fluxonium)

        transmons = layout.get_neighbors(fluxonium)
        for tar_tmon in transmons:
            tar_freq = layout.param("freq", tar_tmon)

            if abs(tar_freq - ctrl_freq_12) < bounds[0]:
                return True

            if abs(tar_freq - ctrl_freq_03) < bounds[0]:
                return True

            if tar_freq < ctrl_freq_12:
                return True

            if tar_freq > ctrl_freq_03:
                return True

            if abs(2 * tar_freq - ctrl_freq_04) < bounds[1]:
                return True

            if abs(3 * tar_freq - ctrl_freq_05) < bounds[2]:
                return True

            for spec_tmon in transmons:
                if spec_tmon != tar_tmon:
                    spec_freq = layout.param("freq", spec_tmon)
                    spec_anharm = layout.param("anharm", spec_tmon)

                    if abs(tar_freq - spec_freq) < bounds[3]:
                        return True

                    spec_12_freq = spec_freq + spec_anharm
                    if abs(tar_freq - spec_12_freq) < bounds[4]:
                        return True

                    if abs(tar_freq + spec_freq - ctrl_freq_04) < bounds[5]:
                        return True
    return False
