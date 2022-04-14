import numpy as np


class ModelTraining:

    def __init__(self, P, M, ext_in, ext_out, val_in, val_out):
        # M and P superior index
        self.P = P
        self.M = M
        # Data loading
        self.extraction_in = ext_in
        self.extraction_out = ext_out
        self.validation_in = val_in
        self.validation_out = val_out

        # Storing coeficients
        self.coef_matrix = self.get_memoryless_model_coef()

    def get_memoryless_model_coef(self):
        # Loading data
        extraction_in = self.extraction_in
        extraction_out = self.extraction_out

        modelled_in = self.model_input(self.P, self.M, extraction_in)
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
