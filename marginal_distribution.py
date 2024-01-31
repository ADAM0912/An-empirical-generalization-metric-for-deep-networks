import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import Rbf
from utils import marginalize
from scipy.optimize import curve_fit, brentq
from scipy.optimize import fsolve

data_all = pd.read_excel(('output_f1f2_distance.xlsx'), header=None)

data_all.columns = ['index', 'randomness','zeroshot','amount','KLg','KLk']

# kl_min = data_all['KL'].min()
# kl_max = data_all['KL'].max()
# data_all['KL'][1:] = 1-((data_all['KL'][1:] - kl_min) / (kl_max - kl_min))

data_for_f1 = data_all[['randomness', 'zeroshot', 'amount', 'KLg']][1:].to_numpy().T
data_for_f2 = data_all[['randomness', 'zeroshot', 'amount', 'KLk']][1:].to_numpy().T
data_pos = data_all[['randomness', 'zeroshot', 'amount']][1:].to_numpy().T


def marginal_pmf(data, variable, probability_column):
    return data.groupby(variable)[probability_column].mean()

marginal_x1 = marginal_pmf(data_all[1:], 'randomness', 'KLg')
marginal_x2 = marginal_pmf(data_all[1:], 'zeroshot', 'KLg')
marginal_x3 = marginal_pmf(data_all[1:], 'amount', 'KLg')

marginal_y1 = marginal_pmf(data_all[1:], 'randomness', 'KLk')
marginal_y2 = marginal_pmf(data_all[1:], 'zeroshot', 'KLk')
marginal_y3 = marginal_pmf(data_all[1:], 'amount', 'KLk')

# def poly2(x, a, b, c):
#     return a * x**2 + b * x + c

def poly2(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def find_intersection(fun1, fun2, x0):
    return fsolve(lambda x : fun1(x) - fun2(x), x0)


#  Randomness

# params_f1, _ = curve_fit(poly2, marginal_x1.index, marginal_x1.values)
# params_f2, _ = curve_fit(poly2, marginal_y1.index, marginal_y1.values)
#
# fitted_f1 = lambda x: poly2(x, *params_f1)
# fitted_f2 = lambda x: poly2(x, *params_f2)
#
# x0_guess = np.mean(data_all[1:]['randomness'])  # Initial guess for intersection
# # intersection_x = find_intersection(fitted_f1, fitted_f2, 0)
# intersection_x = 0.3
# intersection_y1 = fitted_f1(intersection_x)
# intersection_y2 = fitted_f2(intersection_x)
#
# x_range = np.linspace(min(data_all[1:]['randomness']), max(data_all[1:]['randomness']), 100)
#
# plt.figure(figsize=(10, 6))
# plt.scatter(marginal_x1.index, marginal_x1.values, label='Marginal Distributions of Accuracy', color='blue')
# plt.scatter(marginal_y1.index, marginal_y1.values, label='Marginal Distributions of Kappa', color='red')
# plt.plot(x_range, fitted_f1(x_range), label='Fitted Curve for Accuracy', color='blue', linestyle='--')
# plt.plot(x_range, fitted_f2(x_range), label='Fitted Curve for Kappa', color='red', linestyle='--')
#
# # plt.scatter(intersection_x, intersection_y, color='green', marker='o')
# plt.axvline(x=intersection_x, color='green', linestyle='--')
# plt.axhline(y=intersection_y1, color='green', linestyle='--')
# plt.axhline(y=intersection_y2, color='green', linestyle='--')
# # plt.text(intersection_x, 0, f'{intersection_x[0]:.2f}', horizontalalignment='center', verticalalignment='bottom', color='green')
# print(intersection_x, intersection_y1)
# print(intersection_x, intersection_y2)
# plt.xlabel('window size',fontsize=18)
# plt.ylabel('marginal distribution',fontsize=18)
# # plt.title('Marginal Distributions and Fitted Curves for window size with trade-off point',fontsize=18)
# # plt.legend()
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(fontsize=22)
# plt.show()

#  zeroshot
# data_all['zeroshot'] = data_all['zeroshot'] / (50 + data_all['zeroshot'])
#
# marginal_x1 =marginal_pmf(data_all[1:], 'randomness', 'KLg')
# marginal_x2 = marginal_pmf(data_all[1:], 'zeroshot', 'KLg')
# marginal_x3 = marginal_pmf(data_all[1:], 'amount', 'KLg')
#
# marginal_y1 =marginal_pmf(data_all[1:], 'randomness', 'KLk')
# marginal_y2 = marginal_pmf(data_all[1:], 'zeroshot', 'KLk')
# marginal_y3 = marginal_pmf(data_all[1:], 'amount', 'KLk')
#
# params_f1, _ = curve_fit(poly2, marginal_x2.index, marginal_x2.values)
# params_f2, _ = curve_fit(poly2, marginal_y2.index, marginal_y2.values)
#
#
# fitted_f1 = lambda x: poly2(x, *params_f1)
# fitted_f2 = lambda x: poly2(x, *params_f2)
#
# x0_guess = np.mean(data_all[1:]['zeroshot'])  # Initial guess for intersection
# intersection_x = find_intersection(fitted_f1, fitted_f2, x0_guess)
# intersection_x = 20/(20+50)
# intersection_y1 = fitted_f1(intersection_x)
# intersection_y2 = fitted_f2(intersection_x)
#
#
# x_range = np.linspace(min(data_all[1:]['zeroshot']), max(data_all[1:]['zeroshot']), 100)
#
# plt.figure(figsize=(10, 6))
# # marginal_x2.index = [0,0.167,0.286,0.375,0.444,0.500]
# # marginal_y2.index = [0,0.167,0.286,0.375,0.444,0.500]
# plt.scatter(marginal_x2.index, marginal_x2.values, label='Marginal Distributions of Accuracy', color='blue')
# plt.scatter(marginal_y2.index, marginal_y2.values, label='Marginal Distributions of Kappa', color='red')
# plt.plot(x_range, fitted_f1(x_range), label='Fitted Curve for Accuracy', color='blue', linestyle='--')
# plt.plot(x_range, fitted_f2(x_range), label='Fitted Curve for Kappa', color='red', linestyle='--')
#
#
# # plt.scatter(intersection_x, intersection_y, color='green', marker='o')
# plt.axvline(x=intersection_x, color='green', linestyle='--')
# plt.axhline(y=intersection_y1, color='green', linestyle='--')
# plt.axhline(y=intersection_y2, color='green', linestyle='--')
# # plt.text(intersection_x, 0, f'{intersection_x[0]:.2f}', horizontalalignment='center', verticalalignment='bottom', color='green')
#
# plt.xlabel('zeroshot',fontsize=18)
# plt.ylabel('marginal distribution',fontsize=18)
# # plt.title('Marginal Distributions and Fitted Curves for zeroshot with trade-off point',fontsize=18)
# print(intersection_x, intersection_y1)
# print(intersection_x, intersection_y2)
# # plt.legend()
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(fontsize=22)
# plt.show()

#  amount

def poly2(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def poly4(x, a, b, c, d, e,f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e*x +f

def make_poly_function(degree):
    def poly_function(x, *coeffs):
        return sum([coeffs[i] * x**i for i in range(degree + 1)])
    return poly_function

def find_intersection_scan(fun1, fun2, x_values):
    prev_diff = fun1(x_values[0]) - fun2(x_values[0])
    for x in x_values:
        diff = fun1(x) - fun2(x)
        if diff * prev_diff <= 0:  # Sign change indicates a root
            return x
        prev_diff = diff

    return None


def poly6(x, a, b, c, d, e,f,g,h):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e*x +f
    # return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e*x**2 +f*x+g
    # return a * x**7 + b * x**6 + c * x**5 + d * x**4 + e*x**3 +f*x**2+g*x+h
    # return a * x**10 + b * x**9 + c * x**8 + d * x**7 + e*x**6 +f*x**5+g*x**4+h*x**3


# amount_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
amount_list = [0.25e6, 0.5e6, 1e6, 2e6, 4e6, 8e6, 16e6, 32e6]
log_x1 = np.log(amount_list)

params_f1, _ = curve_fit(poly6, log_x1, marginal_x3.values)
params_f2, _ = curve_fit(poly6, log_x1, marginal_y3.values)

fitted_f1 = lambda x: poly6(np.log(x), *params_f1)
fitted_f2 = lambda x: poly6(np.log(x), *params_f2)

x_range = np.linspace(min(log_x1), max(log_x1), 1000)
# Find intersection - need to transform back after finding it in log scale
intersection_log_x = find_intersection_scan(lambda x: poly6(x, *params_f1),
                                       lambda x: poly6(x, *params_f2),
                                       x_range)
intersection_x = np.exp(np.log(8e6))
intersection_y1 = fitted_f1(intersection_x)
intersection_y2 = fitted_f2(intersection_x)
print(intersection_x, intersection_y1)
print(intersection_x, intersection_y2)

# Generate a range of x values for plotting the fitted curves
x_range = np.linspace(min(data_all[1:]['amount']), max(data_all[1:]['amount']), 100)
x_range = np.linspace(min(amount_list), max(amount_list), 100)

# Plot the data, the fitted curves, and the intersection point
plt.figure(figsize=(10, 6))
plt.scatter(amount_list, marginal_x3.values, label='Marginal Distributions of Accuracy', color='blue')
plt.scatter(amount_list, marginal_y3.values, label='Marginal Distributions of Kappa', color='red')
plt.plot(x_range, [fitted_f1(x) for x in x_range], label='Fitted Curve for Accuracy', color='blue', linestyle='--')
plt.plot(x_range, [fitted_f2(x) for x in x_range], label='Fitted Curve for Kappa', color='red', linestyle='--')
# plt.scatter(intersection_x, intersection_y, color='green', marker='o')
plt.axvline(x=intersection_x, color='green', linestyle='--')
plt.axhline(y=intersection_y1, color='green', linestyle='--')
plt.axhline(y=intersection_y2, color='green', linestyle='--')
# plt.text(intersection_x, min(plt.ylim()), f'{intersection_x:.2f}', horizontalalignment='center', verticalalignment='bottom', color='green')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('number of parameters (log scale)',fontsize=18)
plt.ylabel('marginal distribution',fontsize=18)
# plt.title('Marginal Distributions and Fitted Curves for number of parameters with trade-off point',fontsize=16)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=22)
plt.show()