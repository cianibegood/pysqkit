from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy
from os import path

from itertools import count, product
import yaml

STAB_TYPES = ["z_type", "x_type"]


class Layout:
    """
    A general qubit layout class
    """

    def __init__(
        self, layout_setup: Dict,
    ):
        """
        __init__ Initializes the layout class

        Parameters
        ----------
        layout_setup : Dict
            dictionary with the layout name, description and qubit layout.
            The name and description are expected to have string values.
            The qubit layout is expected to be a list of dictionaries.
            Each dictionary defines the name, role,
            stabilizer type, neighbours and transition frequencies.

        Raises
        ------
        NotImplementedError
            If the input arguement is not a dictionary.
        """

        if not isinstance(layout_setup, dict):
            raise ValueError(
                f"layout_setup expected as dict, instead got {type(layout_setup)}"
            )

        self.name = layout_setup.get("name", "")
        self.description = layout_setup.get("description", "")

        self._qubit_info = {}
        self._load_layout(layout_setup)
        self._assign_coords()

    def get_qubits(self, **qubit_params: Any) -> List[str]:
        """
        get_qubits Returns the list of qubits in the layout

        Parameters
        ----------
        **qubit_params : dict, optional
        Extra parameter arguements that can be used to filter the qubit list.
        Refer to Layout.param for the possible values.

        Returns
        -------
        List[str]
            List of qubit names.
        """
        qubit_list = list(self._qubit_info.keys())
        if qubit_params:
            sel_qubits = []
            for qubit in qubit_list:
                try:
                    conds = [
                        self.param(param, qubit) == par_val
                        for param, par_val in qubit_params.items()
                    ]
                    if all(conds):
                        sel_qubits.append(qubit)
                except KeyError:
                    pass
            return sel_qubits
        return qubit_list

    def get_neighbors(
        self, qubit: str, direction: Optional[str] = None, **qubit_params
    ) -> Tuple[str]:
        """
        get_neighbors Returns a list of neighbouring qubits.

        Parameters
        ----------
        qubit : str
            Name of the qubit for which the neighbours are returned.
        direction : Optional[str], optional
            Direction of the neighbour relative to the qubit that is used to filter the list, by default None.
            Possible values are 'north_west', 'north_east', 'south_west', 'south_east'.

        **qubit_params : dict, optional
        Extra parameter arguements that can be used to filter the qubit list.
        Refer to Layout.param for the possible values.

        Returns
        -------
        Tuple[str]
            [description]
        """
        if direction is not None:
            neighors = tuple(self.param("neighbors", qubit)[direction],)
        else:
            neighors = tuple(self.param("neighbors", qubit).values())

        neighors = tuple(filter(lambda qubit: qubit is not None, neighors))

        if qubit_params:
            sel_qubits = []
            for qubit in neighors:
                if all(
                    [
                        self.param(param, qubit) == par_val
                        for param, par_val in qubit_params.items()
                    ]
                ):
                    sel_qubits.append(qubit)
            return tuple(sel_qubits)
        return neighors

    @classmethod
    def from_file(cls, filename: str) -> "Layout":
        """
        from_file Loads the layout class from a .yaml file.

        Returns
        -------
        Layout
            The initialized layout object.

        Raises
        ------
        ValueError
            If the specified file does not exist.
        ValueError
            If the specified file is not a string.
        """
        if isinstance(filename, str):
            if not path.exists(filename):
                raise ValueError("Given path doesn't exist")
        else:
            raise ValueError(
                "Filename must be a string, instead got {}".format(type(filename))
            )

        with open(filename, "r") as setup_file:
            layout_setup = yaml.safe_load(setup_file)
            return cls(layout_setup)

    def param(self, param: str, qubit: str) -> Any:
        """
        param Returns the parameter value of a qubit

        Parameters
        ----------
        param : str
            The name of the qubit parameter.
        qubit : str
            The name of the qubit that is being queried.

        Returns
        -------
        Any
            The value of the parameter
        """
        return self._qubit_info[qubit][param]

    def set_param(self, param: str, qubit: str, value: Any):
        """
        set_param Sets the value of a given qubit parameter

        Parameters
        ----------
        param : str
            The name of the qubit parameter.
        qubit : str
            The name of the qubit that is being queried.
        value : Any
            The new value of the qubit parameter.
        """
        self._qubit_info[qubit][param] = value

    def _load_layout(self, layout_dict):
        """
        _load_layout Internal function that loads the qubit_info dictionary from
        a provided layout dictionary.

        Parameters
        ----------
        layout_dict : dict
            The qubit info dictionary that must be specified in the layout.

        Raises
        ------
        ValueError
            If there are unlabeled qubits in the dictionary.
        ValueError
            If any of the qubits is repeated in the layout.
        """
        chip_layout = deepcopy(layout_dict.get("layout"))
        for qubit_info in chip_layout:
            qubit = qubit_info.pop("qubit", None)

            if qubit is None:
                raise ValueError("Each qubit in the layout must be labeled.")
            if qubit in self._qubit_info:
                raise ValueError("Qubit label repeated, ensure labels are unique.")

            self._qubit_info[qubit] = qubit_info

    def _assign_coords(self):
        """
        _assign_coords Automatically sets the qubit coordinates, if they are not already set

        Parameters
        ----------
        layout : Layout
            The layout of the qubit device.
        """

        qubits = self.get_qubits()
        for qubit in qubits:
            try:
                q_coords = self.param("coords", qubit)
                if q_coords is not None:
                    raise ValueError(
                        "'set_coords' only works on layout where none of the qubits have their coordinates"
                        f" set, instead qubit {qubit} has coordinates {q_coords}"
                    )
            except KeyError:
                pass
        init_qubit = qubits.pop()
        init_cords = (0, 0)

        set_qubits = set()

        def dfs_position(qubit, x_pos, y_pos):
            if qubit not in set_qubits:
                self.set_param("coords", qubit, (x_pos, y_pos))
                set_qubits.add(qubit)

                neighbors_dict = self.param("neighbors", qubit)
                for con_dir, neighbour in neighbors_dict.items():
                    if neighbour:
                        ver_dir, hor_dir = con_dir.split("_")
                        x_shift = -1 if hor_dir == "west" else 1
                        y_shift = -1 if ver_dir == "south" else 1

                        dfs_position(neighbour, x_pos + x_shift, y_pos + y_shift)

        dfs_position(init_qubit, *init_cords)


def surface_code(distance: int, *, mixed_layout: Optional[bool] = True) -> Layout:
    if not isinstance(distance, int):
        raise ValueError("distance provided must be an integer")
    if distance < 0 or (distance % 2) == 0:
        raise ValueError("distance must be an odd positive integer")

    layout_dict = dict(
        name="Distance d={} rotated surface code layout".format(distance),
        description="Layout for a surface code based on a fluxonium-transmon architecture, where the fluxoniums are the ancilla qubits while the transmons are the data qubits.",
    )
    qubit_info_dict = dict()

    if mixed_layout:
        data_type, anc_type = "transmon", "fluxonium"
    else:
        data_type, anc_type = "transmon", "transmon"

    for row_ind, col_ind in product(range(distance), repeat=2):
        qubit = f"D{(row_ind*distance + col_ind) + 1}"
        qubit_info = dict(
            qubit=qubit, role="data", qubit_type=data_type, neighbors=dict(),
        )
        qubit_info_dict[qubit] = qubit_info

    for anc_ind in range(1, int(0.5 * (distance ** 2 - 1)) + 1):
        for stab_type in ["z_type", "x_type"]:
            qubit = f"Z{anc_ind}" if stab_type == "z_type" else f"X{anc_ind}"
            qubit_info = dict(
                qubit=qubit,
                role="anc",
                stab_type=stab_type,
                qubit_type=anc_type,
                neighbors=dict(),
            )
            qubit_info_dict[qubit] = qubit_info

    z_anc_index = count(1)

    for row_ind in range(1, distance):
        for col_ind in range(1 if row_ind % 2 == 0 else 0, distance + 1, 2):
            anc_cord = (row_ind, col_ind)
            anc_qubit = f"Z{next(z_anc_index)}"

            _init_ind = 0 if col_ind == 0 else col_ind - 1
            _end_ind = col_ind if col_ind == distance else col_ind + 1
            row_range = (row_ind - 1, row_ind + 1)
            col_range = (_init_ind, _end_ind)
            neighbors = _data_nighbors(row_range, col_range, anc_cord, distance)

            for data_qubit, data_dir in neighbors:
                qubit_info_dict[anc_qubit]["neighbors"][data_dir] = data_qubit
                anc_dir = _opposite_dir(data_dir)
                qubit_info_dict[data_qubit]["neighbors"][anc_dir] = anc_qubit

    x_anc_index = count(1)

    for row_ind in range(distance + 1):
        for col_ind in range(2 if row_ind % 2 == 0 else 1, distance, 2):
            anc_cord = (row_ind, col_ind)
            anc_qubit = f"X{next(x_anc_index)}"

            _init_ind = 0 if row_ind == 0 else row_ind - 1
            _end_ind = row_ind if row_ind == distance else row_ind + 1
            row_range = (_init_ind, _end_ind)
            col_range = (col_ind - 1, col_ind + 1)
            neighbors = _data_nighbors(row_range, col_range, anc_cord, distance)

            for data_qubit, data_dir in neighbors:
                qubit_info_dict[anc_qubit]["neighbors"][data_dir] = data_qubit
                anc_dir = _opposite_dir(data_dir)
                qubit_info_dict[data_qubit]["neighbors"][anc_dir] = anc_qubit

    layout_dict["layout"] = list(qubit_info_dict.values())
    return Layout(layout_dict)


def _opposite_dir(direction):
    ver_dir, hor_dir = direction.split("_")
    op_ver_dir = "south" if ver_dir == "north" else "north"
    op_hor_dir = "west" if hor_dir == "east" else "east"
    return f"{op_ver_dir}_{op_hor_dir}"


def _data_dir(data_cord, anc_cord):
    data_row, data_col = data_cord
    anc_row, anc_col = anc_cord
    ver_dir = "north" if data_row < anc_row else "south"
    hor_dir = "west" if data_col < anc_col else "east"
    direction = "{}_{}".format(ver_dir, hor_dir)
    return direction


def _data_nighbors(row_range, col_range, anc_cord, dist):
    neighbours = []
    for row_ind in range(*row_range):
        for col_ind in range(*col_range):
            data_qubit = f"D{row_ind*dist + col_ind + 1}"
            direction = _data_dir((row_ind, col_ind), anc_cord)
            neighbours.append((data_qubit, direction))
    return neighbours
