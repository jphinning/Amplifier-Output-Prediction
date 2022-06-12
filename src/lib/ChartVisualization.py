import matplotlib.pyplot as plt
import numpy as np

from lib.OuputPredictor import OutputPredictor


class ChartVisualization (OutputPredictor):

    count_intances = 0

    def __init__(self, P, M, ext_in, ext_out, val_in, val_out, gain, file_name):
        super().__init__(P, M, ext_in, ext_out, val_in, val_out, gain)
        self.gain_increase = gain
        self.file_name = file_name
        self.extraction_AM_AM = self.get_extraction_AM_AM()
        self.validation_AM_AM = self.get_validation_AM_AM()
        self.comparison_val = self.get_comparison_val()
        self.comparison_ext = self.get_comparison_ext()
        ChartVisualization.count_intances += 1

    def get_extraction_AM_AM(self):
        # Plot amplitude for extraction dataset
        self.get_scattered_chart(abs(self.extraction_in), abs(
            self.out_extraction_pred), 'AM-AM Extraction', 'Vout')

    def get_validation_AM_AM(self):
        # Plot amplitude for validation dataset
        self.get_scattered_chart(abs(self.validation_in), abs(
            self.out_validation_pred), 'AM-AM Validation', 'Vout')

    def get_comparison_val(self):
        self.get_scattered_chart(abs(self.validation_in), abs(
            self.out_validation_pred), 'AM-AM Comparison Val', 'Modelado', abs(self.validation_in), abs(
            self.validation_out))

    def get_comparison_ext(self):
        self.get_scattered_chart(abs(self.extraction_in), abs(
            self.out_extraction_pred), 'AM-AM Comparison Ext', 'Modelado', abs(self.extraction_in), abs(
            self.extraction_out))

    def get_scattered_chart(self, input, output, c_title='Data', c_label='data', measured_input=[], measured_output=[]):
        # Ploting charts
        fig, ax = plt.subplots()

        if(len(measured_input) and len(measured_output)):
            plt.plot(measured_input, measured_output, 'o',
                     label="Original", markersize=0.5)

        plt.plot(input, output, 'o', markersize=0.5,
                 label=c_label)

        ax.set(xlabel='input', ylabel='output',
               title=f"{c_title} Gain:{self.gain_increase}")
        plt.legend()
        ax.grid()

        fig.savefig(
            f"{self.file_name}-{c_title}-Gain:{self.gain_increase}-v{ChartVisualization.count_intances}.png")

    def get_NMSE(self, predicted_output, measured_output):
        # NMSE Calculation
        y_ref_sum = 0
        error_sum = 0

        for n in range(len(predicted_output)):
            y_ref_sum += abs(measured_output[n]) ** 2
            error_sum += abs(measured_output[n] - predicted_output[n]) ** 2

        return 10 * np.log10(error_sum/y_ref_sum)
