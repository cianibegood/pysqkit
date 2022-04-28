def address_collision(
    freq_transmon,
    freq_flux_10,
    freq_flux_21,
    freq_flux_32,
    freq_flux_43,
    *,
    delta_i=0.030,
):
    freq_flux_30 = freq_flux_32 + freq_flux_21 + freq_flux_10
    freq_flux_41 = freq_flux_43 + freq_flux_32 + freq_flux_21

    if abs(freq_transmon - freq_flux_21) < delta_i:
        return True
    if abs(freq_transmon - freq_flux_30) < delta_i:
        return True
    if abs(freq_transmon - freq_flux_41) < delta_i:
        return True

    return False


def cross_res_collision(
    target_freq,
    freq_flux_10,
    freq_flux_21,
    freq_flux_32,
    freq_flux_43,
    *,
    delta_i=0.030,
):
    freq_flux_20 = freq_flux_21 + freq_flux_10
    freq_flux_31 = freq_flux_32 + freq_flux_21
    freq_flux_40 = freq_flux_43 + freq_flux_32 + freq_flux_20

    if abs(2 * target_freq - freq_flux_20) < delta_i:
        return True

    if abs(2 * target_freq - freq_flux_31) < delta_i:
        return True

    if abs(2 * target_freq - freq_flux_40) < delta_i:
        return True

    return False


def spectator_collision(
    tar_freq,
    spec_freq,
    spec_anharm,
    *,
    delta_i=0.017,
):
    if abs(tar_freq - spec_freq) < delta_i:
        return True

    if abs(tar_freq - spec_freq - spec_anharm) < delta_i:
        return True

    return False
