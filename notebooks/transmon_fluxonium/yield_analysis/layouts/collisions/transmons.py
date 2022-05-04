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


def cross_res_collision(ctrl_freq, ctrl_aharm, tar_freq, *, delta_cr=0.004):
    if tar_freq > ctrl_freq:
        return True
    if tar_freq < (ctrl_freq + ctrl_aharm):
        return True

    if abs((2 * tar_freq) - (2 * ctrl_freq + ctrl_aharm)) < delta_cr:
        return True

    return False


def spectator_collision(
    ctrl_freq,
    ctrl_anharm,
    tar_freq,
    spec_freq,
    spec_anharm,
    *,
    delta_i=0.017,
    delta_j=0.025,
    delta_k=0.017,
):
    if abs(tar_freq - spec_freq) < delta_i:
        return True

    if abs(tar_freq - spec_freq - spec_anharm) < delta_j:
        return True

    ctrl_02_trans_freq = (2 * ctrl_freq) + ctrl_anharm
    if abs(tar_freq + spec_freq - ctrl_02_trans_freq) < delta_k:
        return True
    return False
