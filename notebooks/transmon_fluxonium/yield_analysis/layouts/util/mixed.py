from typing import List
import numpy as np

from pysqkit.qubits import Fluxonium

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


def set_fluxonium_target_params(
    layout: Layout, charge_energy: float, induct_energy: float, joseph_energy: float,
) -> None:
    fluxoniums = layout.get_qubits(qubit_type="fluxonium")

    for fluxonium in fluxoniums:
        layout.set_param("target_charge_energy", fluxonium, charge_energy)
        layout.set_param("target_induct_energy", fluxonium, induct_energy)
        layout.set_param("target_joseph_energy", fluxonium, joseph_energy)


def sample_params(
    layout: Layout,
    seed: int,
    resist_var: float,
    *,
    num_junctions=100,
    num_fluxonium_levels=6,
):
    rng = np.random.default_rng(seed)

    freq_var = 0.5 * resist_var
    induct_var = resist_var / np.sqrt(num_junctions)
    joseph_var = resist_var

    transmons = layout.get_qubits(qubit_type="transmon")
    for transmon in transmons:
        target_freq = layout.param("target_freq", transmon)
        freq = rng.normal(target_freq, freq_var * target_freq)
        layout.set_param("freq", transmon, freq)

    fluxonia = layout.get_qubits(qubit_type="fluxonium")
    for fluxonium in fluxonia:
        target_induct_energy = layout.param("target_induct_energy", fluxonium)
        induct_energy = rng.normal(
            target_induct_energy, induct_var * target_induct_energy
        )

        target_joseph_energy = layout.param("target_joseph_energy", fluxonium)
        joseph_energy = rng.normal(
            target_joseph_energy, joseph_var * target_joseph_energy
        )

        charge_energy = layout.param("target_charge_energy", fluxonium)

        fluxonium_qubit = Fluxonium(
            label="fluxonium",
            charge_energy=charge_energy,
            induct_energy=induct_energy,
            joseph_energy=joseph_energy,
        )
        fluxonium_qubit.diagonalize_basis(num_fluxonium_levels)

        level_freqs = fluxonium_qubit.eig_energies()
        layout.set_param("freqs", fluxonium, level_freqs)


def get_num_collisions(layout: Layout, bounds: List[float]) -> List[int]:
    num_collisions = np.zeros(8, dtype=int)

    fluxoniums = layout.get_qubits(qubit_type="fluxonium")
    for fluxonium in fluxoniums:
        flux_freqs = layout.param("freqs", fluxonium)

        ctrl_freq_21 = flux_freqs[2] - flux_freqs[1]
        ctrl_freq_30 = flux_freqs[3] - flux_freqs[0]
        ctrl_freq_40 = flux_freqs[4] - flux_freqs[0]
        ctrl_freq_50 = flux_freqs[5] - flux_freqs[0]
        ctrl_freq_51 = flux_freqs[5] - flux_freqs[1]

        transmons = layout.get_neighbors(fluxonium)

        for transmon in transmons:
            tar_freq = layout.param("freq", transmon)

            if abs(tar_freq - ctrl_freq_21) < bounds[0]:
                num_collisions[0] += 1

            if abs(tar_freq - ctrl_freq_30) < bounds[0]:
                num_collisions[0] += 1

            if tar_freq < ctrl_freq_21:
                num_collisions[1] += 1

            if tar_freq > ctrl_freq_30:
                num_collisions[1] += 1

            if abs(2 * tar_freq - ctrl_freq_40) < bounds[1]:
                num_collisions[2] += 1

            if abs(2 * tar_freq - ctrl_freq_51) < bounds[2]:
                num_collisions[3] += 1

            if abs(3 * tar_freq - ctrl_freq_50) < bounds[3]:
                num_collisions[4] += 1

            for spec_qubit in transmons:
                if spec_qubit != transmon:
                    spec_freq = layout.param("freq", spec_qubit)
                    spec_anharm = layout.param("anharm", spec_qubit)

                    if abs(tar_freq - spec_freq) < bounds[4]:
                        num_collisions[5] += 1

                    spec_12_freq = spec_freq + spec_anharm
                    if abs(tar_freq - spec_12_freq) < bounds[5]:
                        num_collisions[6] += 1

                    if abs(tar_freq + spec_freq - ctrl_freq_40) < bounds[6]:
                        num_collisions[7] += 1

    return num_collisions


def any_collisions(layout: Layout, bounds: List[float]) -> List[int]:
    if len(bounds) != 7:
        raise ValueError("Expected only 6 bounds to be provided.")

    fluxoniums = layout.get_qubits(qubit_type="fluxonium")
    for fluxonium in fluxoniums:
        flux_freqs = layout.param("freqs", fluxonium)

        ctrl_freq_21 = flux_freqs[2] - flux_freqs[1]
        ctrl_freq_30 = flux_freqs[3] - flux_freqs[0]
        ctrl_freq_40 = flux_freqs[4] - flux_freqs[0]
        ctrl_freq_50 = flux_freqs[5] - flux_freqs[0]
        ctrl_freq_51 = flux_freqs[5] - flux_freqs[1]

        transmons = layout.get_neighbors(fluxonium)

        for transmon in transmons:
            tar_freq = layout.param("freq", transmon)

            if abs(tar_freq - ctrl_freq_21) < bounds[0]:
                return True

            if abs(tar_freq - ctrl_freq_30) < bounds[0]:
                return True

            if tar_freq < ctrl_freq_21:
                return True

            if tar_freq > ctrl_freq_30:
                return True

            if abs(2 * tar_freq - ctrl_freq_40) < bounds[1]:
                return True

            if abs(2 * tar_freq - ctrl_freq_51) < bounds[2]:
                return True

            if abs(3 * tar_freq - ctrl_freq_50) < bounds[3]:
                return True

            for spec_qubit in transmons:
                if spec_qubit != transmon:
                    spec_freq = layout.param("freq", spec_qubit)
                    spec_anharm = layout.param("anharm", spec_qubit)

                    if abs(tar_freq - spec_freq) < bounds[4]:
                        return True

                    spec_12_freq = spec_freq + spec_anharm
                    if abs(tar_freq - spec_12_freq) < bounds[5]:
                        return True

                    if abs(tar_freq + spec_freq - ctrl_freq_40) < bounds[6]:
                        return True
    return False
