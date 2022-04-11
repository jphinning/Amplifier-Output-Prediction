import numpy as np

from lib.ModelTraining import ModelTraining


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
