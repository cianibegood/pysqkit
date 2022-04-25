from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy
from os import path

import yaml
from matplotlib.pyplot import Axes

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

    def plot(
        self,
        label_qubits: Optional[bool] = True,
        draw_patches: Optional[bool] = False,
        param_label: Optional[str] = None,
        axis: Optional[Axes] = None,
    ):
        """
        plot Plots

        Parameters
        ----------
        label_qubits : Optional[bool], optional
            Whether to label the qubits when plotting them, by default True.
            For codes of larger distance it is recommended to disable this flag.
        draw_patches : Optional[bool], optional
            Whether to draw each stabilizer patch (i.e. color each of the plaquettes), by default False
        axis : Optional[Axes], optional
            matplotlib.pyplot.axi, by default None

        Returns
        -------
        matplotlib.pyplot.figure
            The figure with the plotted layout.
        """
        from .plotter import MatplotlibPlotter

        plotter = MatplotlibPlotter(self, axis)
        return plotter.plot(
            label_qubits, draw_patches=draw_patches, param_label=param_label
        )

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
