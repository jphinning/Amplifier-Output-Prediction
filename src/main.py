from scipy.io import loadmat
import numpy as np

from lib.OuputPredictor import OutputPredictor as op
from lib.ChartVisualization import ChartVisualization as cv

GAIN_1 = 1
GAIN_2 = 1.5


def load_base_data():
    data = loadmat('./src/data/in_out_SBRT2_direto.mat')

    ext_in = np.concatenate(np.array(data['in_extraction']))
    ext_out = np.concatenate(np.array(data['out_extraction']))

    val_in = np.concatenate(np.array(data['in_validation']))
    val_out = np.concatenate(np.array(data['out_validation']))

    return ext_in, val_in, ext_out, val_out


def get_inverse_input(gain):
    ext_in, val_in, ext_out,  val_out = load_base_data()

    P = 2
    M = 0

    model_instance = cv(P, M, ext_in, ext_out, val_in,
                        val_out, gain, 'original')

    md_ext_out = model_instance.out_extraction_md
    md_val_out = model_instance.out_validation_md

    # Create original reversed chart
    model_instance.get_scattered_chart(
        abs(md_ext_out), abs(ext_in), 'Inverse Function Ext', 'Vout')
    model_instance.get_scattered_chart(
        abs(md_val_out), abs(val_in), 'Inverse Function Val', 'Vout')

    return md_ext_out, md_val_out


def get_reverse_chart_gain1(P, M):
    reverse_ext_in, reverse_val_in = get_inverse_input(GAIN_1)
    reverse_ext_out, reverse_val_out, _, _ = load_base_data()

    cv(P, M, reverse_ext_in, reverse_ext_out,
       reverse_val_in, reverse_val_out, GAIN_1, 'inverse')


def get_reverse_chart_gain2(P, M):
    reverse_ext_in, reverse_val_in = get_inverse_input(GAIN_2)
    reverse_ext_out, reverse_val_out, _, _ = load_base_data()

    cv(P, M, reverse_ext_in, reverse_ext_out,
       reverse_val_in, reverse_val_out, GAIN_2, 'inverse')


def main():
    P = int(input("Enter your value for P: "))
    M = 0

    get_reverse_chart_gain1(P, M)
    get_reverse_chart_gain2(P, M)


if __name__ == "__main__":
    main()
