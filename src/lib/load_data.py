from scipy.io import loadmat
import numpy as np


def load_base_data():
    data = loadmat('./src/data/in_out_SBRT2_direto.mat')

    ext_in = np.concatenate(np.array(data['in_extraction']))
    ext_out = np.concatenate(np.array(data['out_extraction']))

    val_in = np.concatenate(np.array(data['in_validation']))
    val_out = np.concatenate(np.array(data['out_validation']))

    return ext_in, val_in, ext_out, val_out
