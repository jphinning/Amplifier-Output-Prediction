import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import lib.complex_plot as cp

data = loadmat('./src/data/in_out_SBRT2_direto.mat')

# print(data)

input_array = np.concatenate(np.array(data['in_extraction']))
output_array = np.concatenate(np.array(data['out_extraction']))

input_validation = np.concatenate(np.array(data['in_validation']))
output_validation = np.concatenate(np.array(data['out_validation']))


gain = input_validation * 1.5


print("Original: ", input_array)
print("GAIN: ", gain)

# cp.complex_plane2(input_array[:100], 1, 'input.png')
# cp.complex_plane2(output_array[:100], 1, 'output.png')
# cp.complex_plane2(input_validation[:100], 1, 'input_validation.png')
# cp.complex_plane2(output_validation[:100], 1, 'output_validation.png')
