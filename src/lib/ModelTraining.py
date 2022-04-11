import numpy as np
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
        P += 1  # add one because of range() in for loops
        M += 1
        regression_matrix_list = []
        # Matrix lines
        for data in range(len(input_array)):
            equation_instance_array = []
            for p in range(1, P):
                for m in range(M):
                    if((data - m) < 0):
                        equation_instance_array.append(
                            (abs(input_array[data]) ** (2*p-2)) * input_array[data])
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
