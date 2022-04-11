import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import lib.get_regression_matrix as rm
import lib.get_linear_combination as lc
import lib.plot_scattered_chart as ps
import lib.NMSE_calc as nmse

# Loading data
data = loadmat('./src/data/in_out_SBRT2_direto.mat')

input_array = np.concatenate(np.array(data['in_extraction']))
output_array = np.concatenate(np.array(data['out_extraction']))

input_validation = np.concatenate(np.array(data['in_validation']))
output_validation = np.concatenate(np.array(data['out_validation']))


# Parameters
p_range = int(input("Enter your value for p: ")) + 1
m_range = int(input("Enter your value for m: ")) + 1

# All lines combined
regression_matrix = rm.modify_input(p_range, m_range, input_array)

# Applying multiple regression
coeficient_array, residuals, rank, s = np.linalg.lstsq(
    regression_matrix, output_array, rcond=None)

# Modifying the input for a new set of data
validation_matrix = rm.modify_input(p_range, m_range, input_validation)

# Predicted output for this new set of data
predicted_output_extraction = lc.multiply_matrix(
    regression_matrix, coeficient_array)

predicted_output_valid = lc.multiply_matrix(
    validation_matrix, coeficient_array)

# Calculating the NMSE
NMSE_validation = nmse.get_NMSE(predicted_output_valid, output_validation)
NMSE_extraction = nmse.get_NMSE(predicted_output_extraction, output_array)
print(f'Extraction : {NMSE_extraction}')
print(f'Validation : {NMSE_validation}')

# Difference between output and input angles
angle_difference = np.angle(output_array, True) - \
    np.angle(input_array, True)

angle_difference_valid = np.angle(output_validation, True) - \
    np.angle(input_validation, True)

angle_difference_pred = np.angle(
    predicted_output_valid, True) - np.angle(input_validation, True)

# Plot amplitude for both datasets
ps.get_scattered_chart(abs(input_array), abs(output_array), 'AM-AM', 'AM-AM')
ps.get_scattered_chart(abs(input_validation), abs(
    output_validation), 'AM-AM_valid', 'AM-AM')


# # Angle difference
# ps.get_scattered_chart(abs(input_validation),
#                        angle_difference_valid, 'AM-PM_valid', 'AM-PM')
# ps.get_scattered_chart(abs(input_array),
#                        angle_difference, 'AM-PM', 'AM-PM')

# Pred Values
ps.get_scattered_chart(abs(input_validation), abs(
    predicted_output_valid), f'AM-AM NMSE:{NMSE_validation:.2f}db', 'Predicted Values', abs(input_validation), abs(
    output_validation))

ps.get_scattered_chart(abs(input_validation), angle_difference_pred,
                       f'AM-PM NMSE:{NMSE_validation:.2f}db', 'Predicted Values', abs(input_validation), angle_difference_valid)
