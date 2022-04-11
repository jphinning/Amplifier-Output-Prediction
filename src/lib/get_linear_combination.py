import numpy as np


def multiply_matrix(input_matrix, coeficient_matrix):
    # Output Calc
    output_array = []
    # Matrix multiplication
    for i in range(len(input_matrix)):
        sum = 0
        for j in range(len(coeficient_matrix)):
            sum += coeficient_matrix[j] * input_matrix[i][j]

        output_array.append(sum)

    output = np.array(output_array)
    return output
