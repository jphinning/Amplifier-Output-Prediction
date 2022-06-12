
from lib.OuputPredictor import OutputPredictor as op
from lib.ChartVisualization import ChartVisualization as cv
from lib.load_data import load_base_data

GAIN_1 = 1
GAIN_2 = 1.5
GAIN_3 = 0.65

class ReverseChartModel ():
  def _get_inverse_input(self, gain):
    ext_in, val_in, ext_out, val_out = load_base_data()

    P = 2
    M = 1

    model_instance = cv(P, M, ext_in, ext_out, val_in,
                        val_out, gain, 'org')

    md_ext_out = model_instance.out_extraction_pred
    md_val_out = model_instance.out_validation_pred

    return md_ext_out, md_val_out
  
  def _get_reverse_chart(self, P, M, gain):
      reverse_ext_in, reverse_val_in = self._get_inverse_input(gain)
      reverse_ext_out, reverse_val_out, _, _ = load_base_data()

      cv(P, M, reverse_ext_in, reverse_ext_out,
        reverse_val_in, reverse_val_out, 1, 'inv')


  def get_modelled_chart(self):
      P = int(input("Enter your value for P: "))
      M = int(input("Enter your value for M: "))

      # get_reverse_chart(P, M, GAIN_1)
      # get_reverse_chart(P, M, GAIN_2)
      self._get_reverse_chart(P, M, GAIN_3)


# test = ReverseChartModel().get_modelled_chart()