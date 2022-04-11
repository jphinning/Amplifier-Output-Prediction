from math import log10


def get_NMSE(predicted_output, measured_output):

    y_ref_sum = 0
    error_sum = 0

    for n in range(len(predicted_output)):
        y_ref_sum += abs(measured_output[n]) ** 2
        error_sum += abs(measured_output[n] - predicted_output[n]) ** 2

    return 10 * log10(error_sum/y_ref_sum)
