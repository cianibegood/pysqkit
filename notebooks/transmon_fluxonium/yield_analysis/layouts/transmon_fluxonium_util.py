from typing import List
import numpy as np

from pysqkit.qubits import Fluxonium, fluxonium

from .layout import Layout
from .transmon_fluxonium_collisions import (
    address_collision,
    cross_res_collision,
    spectator_collision,
)


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

    fluxonia = layout.get_qubits(qubit_type="fluxonium")
    for fluxonium in fluxonia:
        layout.set_param("freq_group", fluxonium, 0)


def set_transmon_target_freqs(
    layout: Layout, group_freqs: List[float], group_anharms: List[float]
) -> None:
    if len(group_freqs) != 4:
        raise ValueError(
            "5 distinct qubit frequencies are required for the square layout."
        )
    qubits = layout.get_qubits(qubit_type="transmon")

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

        if freq_group == 0:
            raise ValueError("Only fluxonia can have frequency group 0")

        target_freq = group_freqs[freq_group - 1]
        layout.set_param("target_freq", qubit, target_freq)

        anharm = group_anharms[freq_group - 1]
        layout.set_param("anharm", qubit, anharm)


def set_fluxonia_target_energies(
    layout: Layout, target_induct_energy: float, target_joseph_energy: float
) -> None:
    qubits = layout.get_qubits(qubit_type="fluxonium")

    for qubit in qubits:
        layout.set_param("target_induct_energy", qubit, target_induct_energy)
        layout.set_param("target_joseph_energy", qubit, target_joseph_energy)


def sample_params(
    layout: Layout,
    seed: int,
    resist_var: float,
    *,
    charge_energy=1,
    num_junctions=100,
    num_fluxonium_levels=5,
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

        fluxonium_qubit = Fluxonium(
            label="fluxonium",
            charge_energy=charge_energy,
            induct_energy=induct_energy,
            joseph_energy=joseph_energy,
        )
        fluxonium_qubit.diagonalize_basis(num_fluxonium_levels)

        energies = fluxonium_qubit.eig_energies()
        for state_i, state_j in [(1, 0), (2, 1), (3, 2), (4, 3)]:
            trans_freq = energies[state_i] - energies[state_j]
            layout.set_param(f"freq_{state_i}{state_j}", fluxonium, trans_freq)


def get_collisions(layout: Layout) -> List[int]:
    num_address_collisions = 0
    num_cross_res_collisions = 0
    num_spectator_collisions = 0

    fluxonia = layout.get_qubits(qubit_type="fluxonium")
    for fluxonium in fluxonia:
        freq_10 = layout.param("freq_10", fluxonium)
        freq_21 = layout.param("freq_21", fluxonium)
        freq_32 = layout.param("freq_32", fluxonium)
        freq_43 = layout.param("freq_43", fluxonium)

        transmons = layout.get_neighbors(fluxonium)

        for transmon in transmons:
            transmon_freq = layout.param("freq", transmon)
            # transmon_anharm = layout.param("anharm", transmon)

            if address_collision(transmon_freq, freq_10, freq_21, freq_32, freq_43):
                num_address_collisions += 1

            if cross_res_collision(transmon_freq, freq_10, freq_21, freq_32, freq_43):
                num_cross_res_collisions += 1

            for spectator in transmons:
                if spectator != transmon:
                    spectator_freq = layout.param("freq", spectator)
                    spectator_anharm = layout.param("anharm", spectator)

                    if spectator_collision(
                        transmon_freq,
                        spectator_freq,
                        spectator_anharm,
                    ):
                        num_spectator_collisions += 1
    result = np.array(
        [num_address_collisions, num_cross_res_collisions, num_spectator_collisions],
        dtype=int,
    )
    return result
