import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import griddata


# Define your custom problem
data_all = pd.read_excel(('output_clip_cifar.xlsx'), header=None)
data_all.columns = ['randomness', 'zeroshot', 'amount', 'f1', 'f2','f3', 'f4','f5','f6']
values_to_remove = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
data_all = data_all[~data_all['amount'].isin(values_to_remove)]
# Lists as provided
randomness = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
zeroshot = [0, 10, 20, 30, 40, 50]
# zeroshot = [0,0.167,0.286,0.375,0.444,0.500]
# amount = [64, 128, 256, 512, 1024]
model_name = ['RN50', 'RN101', 'RN50x4', 'RN50x16','RN50x64']
amount = [38, 56, 87, 167, 420]
zeroshot_percent = [0, 0.167, 0.286, 0.375, 0.444, 0.500]


def normalize_amount(array):
    # Calculate the minimum and maximum values of the array
    min_val = min(array)
    max_val = max(array)

    # Apply the normalization formula to each element in the array
    normalized_array = [(x - min_val) / (max_val - min_val) for x in array]

    return normalized_array

def denormalize_value(y):
    min_val = 38
    max_val = 420

    # Convert normalized value y back to the original range
    x = y * (max_val - min_val) + min_val
    return x

# Example usage
normalized_values = normalize_amount(amount)

for i in range(len(model_name)):
    data_all['amount'] = data_all['amount'].replace(model_name[i], normalized_values[i])

for i in range(len(zeroshot_percent)):
    data_all['zeroshot'] = data_all['zeroshot'].replace(zeroshot[i], zeroshot_percent[i])



def value_to_number(y):
    m = 0.010471204188481676  # Slope from previous calculation
    b = 0.6020942408376964  # Intercept from previous calculation

    # Calculate x from y
    x = (y - b) / m

    # Ensure x is between 38 and 420
    if x < 38:
        return x+0.0001
    elif x > 420:
        return x-0.0001
    else:
        return x

# print(value_to_number(1))
data_all = data_all[1:]
# data_all['randomness'] = 1-data_all['randomness']
# data_all['amount'] = 1-data_all['amount']
data_all['zeroshot'] = 1-data_all['zeroshot']

def idw_interpolation_3d_grid(x, y, z, values, xi, yi, zi, power=2):
    """
    Perform 3D IDW interpolation on a structured grid.

    Parameters:
        x (1D array): 1D array of x-coordinates (size n_x).
        y (1D array): 1D array of y-coordinates (size n_y).
        z (1D array): 1D array of z-coordinates (size n_z).
        values (3D array): 3D array of values of shape (n_x, n_y, n_z).
        xi (float): x-coordinate of the point to interpolate.
        yi (float): y-coordinate of the point to interpolate.
        zi (float): z-coordinate of the point to interpolate.
        power (int): power parameter which controls the decay of influence with distance.

    Returns:
        float: interpolated value at point (xi, yi, zi).
    """
    # Create a mesh grid of x, y, z coordinates
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    # Flatten the grid coordinates and values
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    flat_z = grid_z.ravel()
    flat_values = values.ravel()

    # Calculate the distances from each grid point to the interpolation point
    distances = np.sqrt(((xi - flat_x) ** 2 + (yi - flat_y) ** 2 + (zi - flat_z) ** 2).astype(float))
    # Avoid division by zero
    distances = np.where(distances == 0, 1e-12, distances)
    # Calculate weights using inverse distance weighting
    weights = 1 / distances ** power
    # Calculate the interpolated value
    interpolated_value = np.sum(weights * flat_values) / np.sum(weights)

    return interpolated_value

# Unique values for each variable
randomness_values = np.array(data_all['randomness'].unique())
zeroshot_values = np.array(data_all['zeroshot'].unique())
# amount_values = sorted(data_all['amount'].unique())
amount_values =  np.array(data_all['amount'].unique())

# randomness_values = sorted(data_all['randomness'].unique())
# zeroshot_values = sorted(data_all['zeroshot'].unique())
# amount_values =  sorted(data_all['amount'].unique())
# Prepare values for interpolation
def prepare_values_for_interpolation(column_name):
    # Reshape the data to match the grid
    reshaped_data = data_all.pivot_table(index='randomness', columns=['zeroshot', 'amount'], values=column_name).values
    reshaped_data = reshaped_data.reshape(len(randomness_values), len(zeroshot_values), len(amount_values))
    return reshaped_data

f1_values = prepare_values_for_interpolation('f1')
f2_values = prepare_values_for_interpolation('f2')
f3_values = prepare_values_for_interpolation('f3')
f4_values = prepare_values_for_interpolation('f4')
f5_values = prepare_values_for_interpolation('f5')
f6_values = prepare_values_for_interpolation('f6')

def interpolate_f1_f2(x1, x2, x3):
    # randomness = 0.6+(x1-1)*0.05
    # zeroshot   = (x2-1)*10
    # # zeroshot = (x2) * 10
    # amount = value_to_number(x3)


    # f1_interp= idw_interpolation_3d_grid(randomness_values, zeroshot_values, amount_values, f1_values, x1, x2, x3)
    # f2_interp = idw_interpolation_3d_grid(randomness_values, zeroshot_values, amount_values, f2_values, x1, x2, x3)
    # f3_interp = idw_interpolation_3d_grid(randomness_values, zeroshot_values, amount_values, f3_values, x1, x2, x3)
    # f4_interp = idw_interpolation_3d_grid(randomness_values, zeroshot_values, amount_values, f4_values, x1, x2, x3)
    # f5_interp = idw_interpolation_3d_grid(randomness_values, zeroshot_values, amount_values, f5_values, x1, x2, x3)
    # f6_interp = idw_interpolation_3d_grid(randomness_values, zeroshot_values, amount_values, f6_values, x1, x2, x3)
    points = np.array((data_all['randomness'],data_all['zeroshot'],data_all['amount'])).T
    request = np.array([[x1, x2, x3]])
    f1_interp = griddata(points, data_all['f1'],request)
    f2_interp = griddata(points, data_all['f2'], request)
    f3_interp = griddata(points, data_all['f3'], request)
    f4_interp = griddata(points, data_all['f4'], request)
    f5_interp = griddata(points, data_all['f5'], request)
    f6_interp = griddata(points, data_all['f6'], request)
    return f1_interp, f2_interp, f3_interp, f4_interp, f5_interp, f6_interp




def objective_function(vars):
    # Assuming interpolate_f1_f2 is defined elsewhere
    x, y, z, c1, c2, c3 = vars
    error, kappa, std_error, std_kappa, percentile_error, percentile_kappa = interpolate_f1_f2(x, y, z)
    # return 1 - acc + kappa
    output = error + kappa + std_error + std_kappa + percentile_error + percentile_kappa
    # a1=1
    # a2=0.75
    # a3=0.5
    # imagenet best 0.8 0.25 0.25
    # a1=0.80
    # a2=0.25
    # a3=0.25

    # cifar
    a1=0.285
    a2=0.8
    a3=0.8
    return output + a1*c1**2 + a2*c2**2 + a3*c3**2


def constraint1(vars):
    x, y, z, c1, c2, c3 = vars
    return c1 - x

def constraint2(vars):
    x, y, z, c1, c2, c3 = vars
    return c2 - y

def constraint3(vars):
    x, y, z, c1, c2, c3 = vars
    return c3 - z

x0 = [0.6, 0.5, 0, 1, 1, 1]

bnds = [(0.6, 1), (0.5, 1), (0, 1), (0.6, 1), (0.5, 1), (0, 1)]

cons = [{'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3}]


solution = minimize(objective_function, x0, method='SLSQP', bounds=bnds, constraints=cons)
x, y, z, c1, c2, c3 = solution.x

print(f'Optimal solution: x = {x}, y = {1-y}, z = {denormalize_value(z)}, c1 = {c1}, c2 = {c2}, c3 = {c3}')
#

