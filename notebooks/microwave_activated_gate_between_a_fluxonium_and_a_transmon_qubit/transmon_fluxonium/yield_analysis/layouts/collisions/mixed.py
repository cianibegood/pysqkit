def address_collision(freq_transmon, freqs_fluxonium, *, delta_i=0.017):
    freq_fluxonium_21 = freqs_fluxonium[2] - freqs_fluxonium[1]
    freq_fluxonium_30 = freqs_fluxonium[3] - freqs_fluxonium[0]
    freq_fluxonium_41 = freqs_fluxonium[4] - freqs_fluxonium[1]

    if abs(freq_transmon - freq_fluxonium_21) < delta_i:
        return True
    if abs(freq_transmon - freq_fluxonium_30) < delta_i:
        return True
    # Next one can be prob disabled
    if abs(freq_transmon - freq_fluxonium_41) < delta_i:
        return True

    return False


def cross_res_collision(freq_transmon, freqs_fluxonium, *, delta_i=0.05, delta_j=0.1):

    freq_fluxonium_20 = freqs_fluxonium[2] - freqs_fluxonium[0]
    freq_fluxonium_21 = freqs_fluxonium[2] - freqs_fluxonium[1]
    freq_fluxonium_30 = freqs_fluxonium[3] - freqs_fluxonium[0]
    freq_fluxonium_31 = freqs_fluxonium[3] - freqs_fluxonium[1]
    freq_fluxonium_40 = freqs_fluxonium[4] - freqs_fluxonium[0]

    if freq_transmon < freq_fluxonium_21 or freq_transmon > freq_fluxonium_30:
        return True

    if abs(freq_transmon - freq_fluxonium_21) < delta_j:
        return True

    if abs(freq_transmon - freq_fluxonium_30) < delta_j:
        return True

    if abs(2 * freq_transmon - freq_fluxonium_20) < delta_i:
        return True

    if abs(2 * freq_transmon - freq_fluxonium_31) < delta_i:
        return True

    if abs(2 * freq_transmon - freq_fluxonium_40) < delta_i:
        return True

    return False


def spectator_collision(tar_freq, spectator_freq, spectator_anharm, *, delta_i=0.02):
    if abs(tar_freq - spectator_freq) < delta_i:
        return True

    if abs(tar_freq - spectator_freq - spectator_anharm) < delta_i:
        return True

    return False
