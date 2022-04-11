import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


class ModelTraining:

    def __init__(self):
        self.data = loadmat('./src/data/in_out_SBRT2_direto.mat')
        self.extraction_in = np.concatenate(
            np.array(self.data['in_extraction']))
        self.extraction_out = np.concatenate(
            np.array(self.data['out_extraction']))
        self.validation_in = np.concatenate(
            np.array(self.data['in_validation']))
        self.validation_out = np.concatenate(
            np.array(self.data['out_validation']))
        self.coef_matrix = self.get_memoryless_model_coef()

    def get_memoryless_model_coef(self):
        # Loading data
        extraction_in = self.extraction_in
        extraction_out = self.extraction_out

        # M and P superior index
        P = 3
        M = 0

        modelled_in = self.model_input(P, M, extraction_in)
        coef_array = self.get_coeficients(modelled_in, extraction_out)

        return coef_array

    def model_input(self, P, M, input_array):
        P += 1  # add one because of the range in for loops
        M += 1
        regression_matrix_list = []
        # Matrix lines
        for data in range(len(input_array)):
            equation_instance_array = []
            for p in range(1, P):
                for m in range(M):
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

    def get_coeficients(self, modelled_input, output_array):
        # Applying multiple regression
        coeficient_array, residuals, rank, s = np.linalg.lstsq(
            modelled_input, output_array, rcond=None)

        return coeficient_array


class OutputPredictor(ModelTraining):

    def __init__(self, gain):
        super().__init__()
        # Modelled outputs for memoryless polinomial
        self.out_extraction_md = self.get_pred_out_extraction(gain)
        self.out_validation_md = self.get_pred_out_validation(gain)

    def get_pred_out_extraction(self, gain):
        modelled_extraction = self.model_input(
            3, 0, gain * self.extraction_in)

        out_extraction = self.multiply_matrix(
            modelled_extraction, self.coef_matrix)

        return out_extraction

    def get_pred_out_validation(self, gain):
        modelled_validation = self.model_input(
            3, 0, gain * self.validation_in)

        out_validation = self.multiply_matrix(
            modelled_validation, self.coef_matrix)

        return out_validation

    def multiply_matrix(self, modelled_input, coeficient_matrix):
        # Output Calc
        output_array = []
        # Matrix multiplication
        for i in range(len(modelled_input)):
            sum = 0
            for j in range(len(coeficient_matrix)):
                sum += coeficient_matrix[j] * modelled_input[i][j]

            output_array.append(sum)

        output = np.array(output_array)

        return output


class ChartVisualization (OutputPredictor):

    def __init__(self, gain):
        super().__init__(gain)

    def get_extraction_AM_AM(self):
        # Plot amplitude for extraction dataset
        self.get_scattered_chart(abs(self.extraction_in), abs(
            self.out_extraction_md), 'AM-AM Extraction', 'AM-AM')

    def get_validation_AM_AM(self):
        # Plot amplitude for validation dataset
        self.get_scattered_chart(abs(self.validation_in), abs(
            self.out_validation_md), 'AM-AM Validation', 'AM-AM')

    def get_scattered_chart(self, input, output, c_title='Data', c_label='data', measured_input=[], measured_output=[]):
        # Ploting charts
        fig, ax = plt.subplots()

        if(len(measured_input) and len(measured_output)):
            plt.plot(measured_input, measured_output, 'o',
                     label="Measured Values", markersize=0.5)

        plt.plot(input, output, 'o',
                 label=c_label, markersize=0.5)

        ax.set(xlabel='input', ylabel='output', title=c_title)
        plt.legend()
        ax.grid()

        fig.savefig(f"{c_title}.png")

    def get_NMSE(self, predicted_output, measured_output):
        # NMSE Calculation
        y_ref_sum = 0
        error_sum = 0

        for n in range(len(predicted_output)):
            y_ref_sum += abs(measured_output[n]) ** 2
            error_sum += abs(measured_output[n] - predicted_output[n]) ** 2

        return 10 * np.log10(error_sum/y_ref_sum)
