import numpy as np


def modify_input(p_range, m_range, input_array):
    regression_matrix_list = []
    # Matrix lines
    for data in range(len(input_array)):
        equation_instance_array = []
        for p in range(1, p_range):
            for m in range(m_range):
                if((data - m) < 0):
                    equation_instance_array.append(
                        (abs(input_array[data])) ** (2*p-2) * input_array[data])
                else:
                    equation_instance_array.append(
                        (abs(input_array[data - m]) ** (2*p-2)) * input_array[data - m])

        regression_matrix_list.append(equation_instance_array)

    # All lines combined
    regression_matrix = np.array(regression_matrix_list)
    return regression_matrix
