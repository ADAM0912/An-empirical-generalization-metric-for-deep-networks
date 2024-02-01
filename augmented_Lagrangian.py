import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import numpy as np
from scipy.optimize import minimize

# Define your custom problem
data_all = pd.read_excel(('output_f1f2_distance.xlsx'), header=None)
data_all.columns = ['index', 'randomness', 'zeroshot', 'amount', 'f1', 'f2']

# Lists as provided
randomness = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
zeroshot = [0, 10, 20, 30, 40, 50]
amount = [64, 128, 256, 512, 1024, 2048, 4096, 8192]




data_all = data_all[1:]

# index_to_drop = data_all[(data_all['randomness'] == 0) &
#                          (data_all['zeroshot'] == 0) &
#                          (data_all['amount'] == 8192)].index
#
# data_all[(data_all['randomness'] == 0) &
#                          (data_all['zeroshot'] == 0) &
#                          (data_all['amount'] == 8192)]['f1'].data=0.01
# data_all[(data_all['randomness'] == 0) &
#                          (data_all['zeroshot'] == 0) &
#                          (data_all['amount'] == 8192)]['f2'].value=0.01

# Unique values for each variable
randomness_values = sorted(data_all['randomness'].unique())
zeroshot_values = sorted(data_all['zeroshot'].unique())
amount_values = sorted(data_all['amount'].unique())

# Prepare values for interpolation
def prepare_values_for_interpolation(column_name):
    # Reshape the data to match the grid
    reshaped_data = data_all.pivot_table(index='randomness', columns=['zeroshot', 'amount'], values=column_name).values
    reshaped_data = reshaped_data.reshape(len(randomness_values), len(zeroshot_values), len(amount_values))
    return reshaped_data

f1_values = prepare_values_for_interpolation('f1')
f2_values = prepare_values_for_interpolation('f2')

# Interpolation functions
f1_interpolator = RegularGridInterpolator((randomness_values, zeroshot_values, amount_values), f1_values)
f2_interpolator = RegularGridInterpolator((randomness_values, zeroshot_values, amount_values), f2_values)

# Interpolation function
def interpolate_f1_f2(x1, x2, x3):
    randomness = x1/10-0.1
    zeroshot   = (x2-1)*10
    amount = np.exp2(x3+5)


    point = np.array([randomness, zeroshot, amount])
    f1_interp = f1_interpolator(point)
    f2_interp = f2_interpolator(point)
    return f1_interp, f2_interp



def objective_function(x):
    # Assuming interpolate_f1_f2 is defined elsewhere
    acc, kappa = interpolate_f1_f2(x[0], x[1], x[2])
    return 1 - acc + kappa


def equality_constraint(x):
    # Constraint of the form A*x = b
    return np.dot(np.array([1, 1, 1]), x) - np.array([10, 6, 8])


def augmented_lagrangian(x, lambd, rho):
    # Objective function
    obj = objective_function(x)

    # Augmented Lagrangian for equality constraints
    eq_constraint = equality_constraint(x)
    lagrangian_eq = lambd.dot(eq_constraint) + (rho / 2) * np.sum(eq_constraint ** 2)

    # Total Augmented Lagrangian
    L = obj + lagrangian_eq

    return L


def solve_augmented_lagrangian(initial_x, initial_lambda, rho, max_iter=100):
    x = initial_x
    lambd = initial_lambda

    for i in range(max_iter):
        # Minimize the augmented Lagrangian
        res = minimize(lambda x: augmented_lagrangian(x, lambd, rho), x, method='SLSQP',
                       bounds=[(1, 10), (1, 6), (1, 8)])
        x = res.x

        # Update the Lagrange multipliers
        eq_constraint = equality_constraint(x)
        lambd = lambd + rho * eq_constraint

        # Optionally, update rho and check for convergence here

    return x


# Initial guess and parameters
initial_x = np.array([1, 1, 1])
initial_lambda = np.array([0, 0, 0])
# rho = 0.02149454
rho = 0.03

# Solve the problem
optimal_x = solve_augmented_lagrangian(initial_x, initial_lambda, rho)
print("Optimal x:", optimal_x)




