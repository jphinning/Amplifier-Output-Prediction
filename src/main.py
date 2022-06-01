from matplotlib import pyplot as plt
import numpy as np
from lib.ModelTraining import ModelTraining as mt
# from lib.OuputPredictor import OutputPredictor as op
# from lib.ChartVisualization import ChartVisualization as cv
from lib.load_data import load_base_data

GAIN_1 = 1
GAIN_2 = 1.5

ext_in, val_in, ext_out, val_out = load_base_data()


def log(str):
    text_file = open("log.txt", "a")
    n = text_file.write(str)
    text_file.close()


def model_oneToOne_input(P, M, input_array, gain):
    P += 1  # add one because of range() in for loops
    M += 1
    input_array *= gain
    ext = ext_in * gain
    regression_matrix_list = []
    # Matrix lines
    for data in range(len(input_array)):
        equation_instance_array = []
        for p in range(1, P):
            for m in range(M):
                if((data - m) < 0):
                    equation_instance_array.append(
                        (abs(input_array[data]) ** (2*p-2)) * input_array[data])
                elif(m >= 1):
                    equation_instance_array.append(
                        (abs(ext[data - m]) ** (2*p-2)) * ext[data - m])
                else:
                    equation_instance_array.append(
                        (abs(input_array[data]) ** (2*p-2)) * input_array[data])

        regression_matrix_list.append(equation_instance_array)

    # All lines combined
    regression_matrix = np.array(regression_matrix_list)
    return regression_matrix


def model_manual_at_n(gain, n):
    P = 3  # add one because of range() in for loops
    M = 2

    ext = ext_in * gain
    max_val = ext[n]
    input_array = np.linspace(0, max_val, num=100)

    regression_matrix_list = []
    # Matrix lines
    for data in range(len(input_array)):
        equation_instance_array = []
        for p in range(1, P):
            for m in range(M):
                if((data - m) < 0):
                    equation_instance_array.append(
                        (abs(input_array[data]) ** (2*p-2)) * input_array[data])
                elif(m >= 1):
                    equation_instance_array.append(
                        (abs(ext[n - m]) ** (2*p-2)) * ext[n - m])
                else:
                    equation_instance_array.append(
                        (abs(input_array[data]) ** (2*p-2)) * input_array[data])

        regression_matrix_list.append(equation_instance_array)

    # All lines combined
    regression_matrix = np.array(regression_matrix_list)
    return regression_matrix, input_array


def scattered_chart(input, output, c_title='Data', c_label='data', measured_input=[], measured_output=[]):
    # Ploting charts
    fig, ax = plt.subplots()

    if(len(measured_input) and len(measured_output)):
        plt.plot(measured_input, measured_output, 'o',
                 label="Original", markersize=0.5)

    plt.plot(input, output, 'o', markersize=1,
             label=c_label)

    ax.set(xlabel='input', ylabel='output',
           title=f"{c_title}")
    plt.legend()
    ax.grid()

    fig.savefig(
        f"{c_title}-Gain:1.png")


def main():

    # P = int(input("Enter your value for P: "))
    # M = int(input("Enter your value for M: "))
    P = 2
    M = 1

    org_poly = mt(P, M, ext_in, ext_in, val_in, val_out)
    print(org_poly.coef_matrix)
    # print(org_poly.coef_matrix)
    # data1 = model_oneToOne_input(P, M, ext_in, GAIN_1)
    # data2 = model_oneToOne_input(P, M, val_in, GAIN_1)
    # data3 = model_oneToOne_input(P, M, ext_in, GAIN_2)
    # data4 = model_oneToOne_input(P, M, val_in, GAIN_2)

    for n in range(16):
        data_manual1, input1 = model_manual_at_n(1, n)
        output_manual1 = data_manual1 @ org_poly.coef_matrix
        scattered_chart(input1, output_manual1, f"n={n}")

    # output1 = data1 @ org_poly.coef_matrix
    # output2 = data2 @ org_poly.coef_matrix
    # output3 = data3 @ org_poly.coef_matrix
    # output4 = data4 @ org_poly.coef_matrix

    # scattered_chart(ext_in, output1, "ext_in g1")
    # scattered_chart(val_in, output2, "val_in g1")
    # scattered_chart(ext_in, output3, "ext_in g2")
    # scattered_chart(val_in, output4, "val_in g2")


if __name__ == "__main__":
    main()
