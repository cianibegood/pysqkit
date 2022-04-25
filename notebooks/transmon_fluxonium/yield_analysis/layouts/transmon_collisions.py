def address_collision(
    freq_i, anharm_i, freq_j, anharm_j, *, delta_i=0.017, delta_j=0.030
):
    if abs(freq_i - freq_j) < delta_i:
        return True
    if abs(freq_i - freq_j - anharm_j) < delta_j:
        return True
    if abs(freq_j - freq_i - anharm_i) < delta_j:
        return True

    return False


def cross_res_collision(ctrl_freq, ctrl_aharm, target_freq, *, delta_cr=0.004):
    if target_freq > ctrl_freq:
        return True
    if target_freq < (ctrl_freq + ctrl_aharm):
        return True

    if abs((2 * target_freq) - (2 * ctrl_freq) - ctrl_aharm) < delta_cr:
        return True

    return False


def spectator_collision(
    control_freq,
    control_anharm,
    target_freq,
    spectator_freq,
    spectator_anharm,
    *,
    delta_i=0.017,
    delta_j=0.025,
    delta_k=0.017,
):
    if abs(target_freq - spectator_freq) < delta_i:
        return True

    if abs(target_freq - spectator_freq - spectator_anharm) < delta_j:
        return True

    if (
        abs(target_freq + spectator_freq - (2 * control_freq) - control_anharm)
        < delta_k
    ):
        return True
    return False

