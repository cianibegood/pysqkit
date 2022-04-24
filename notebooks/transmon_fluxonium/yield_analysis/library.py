from itertools import count, product

from .layout import Layout

STAB_TYPES = ["z_type", "x_type"]


def surface_code(distance: int) -> Layout:
    if not isinstance(distance, int):
        raise ValueError("distance provided must be an integer")
    if distance < 0 or (distance % 2) == 0:
        raise ValueError("distance must be an odd positive integer")

    layout_dict = dict(
        name="Distance d={} rotated surface code layout".format(distance),
        stab_order={
            "x_type": ["north_east", "north_west", "south_east", "south_west"],
            "z_type": ["north_east", "south_east", "north_west", "south_west"],
        },
    )
    qubit_info_dict = dict()

    for row_ind, col_ind in product(range(distance), repeat=2):
        qubit = f"D{(row_ind*distance + col_ind) + 1}"
        qubit_info = dict(
            qubit=qubit,
            role="data",
            neighbors=dict(),
        )
        qubit_info_dict[qubit] = qubit_info

    for anc_ind in range(1, int(0.5 * (distance ** 2 - 1)) + 1):
        for stab_type in ["z_type", "x_type"]:
            qubit = f"Z{anc_ind}" if stab_type == "z_type" else f"X{anc_ind}"
            qubit_info = dict(
                qubit=qubit,
                role="anc",
                stab_type=stab_type,
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
            # anc_qubit = f"Z{next(x_anc_index)}"

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
