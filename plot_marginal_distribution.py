import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit, brentq
from scipy.optimize import fsolve

data_all = pd.read_excel(('output_eff_cifar.xlsx'), header=None)

data_all.columns = ['ssim','zeroshot','model_type','error','kappa','Std Error','Std Kappa', 'percentile_error', 'percentile_kappa']
data_pos = data_all[['ssim', 'zeroshot', 'model_type']][1:].to_numpy().T


def marginal_pmf(data, variable, probability_column):
    return data.groupby(variable)[probability_column].mean()

marginal_x1 = marginal_pmf(data_all[1:], 'ssim', 'error')
marginal_x2 = marginal_pmf(data_all[1:], 'zeroshot', 'error')
marginal_x3 = marginal_pmf(data_all[1:], 'model_type', 'error')

marginal_y1 = marginal_pmf(data_all[1:], 'ssim', 'kappa')
marginal_y2 = marginal_pmf(data_all[1:], 'zeroshot', 'kappa')
marginal_y3 = marginal_pmf(data_all[1:], 'model_type', 'kappa')

marginal_a1 = marginal_pmf(data_all[1:], 'ssim', 'Std Error')
marginal_a2 = marginal_pmf(data_all[1:], 'zeroshot', 'Std Error')
marginal_a3 = marginal_pmf(data_all[1:], 'model_type', 'Std Error')

marginal_b1 = marginal_pmf(data_all[1:], 'ssim', 'Std Kappa')
marginal_b2 = marginal_pmf(data_all[1:], 'zeroshot', 'Std Kappa')
marginal_b3 = marginal_pmf(data_all[1:], 'model_type', 'Std Kappa')

marginal_c1 = marginal_pmf(data_all[1:], 'ssim', 'percentile_error')
marginal_c2 = marginal_pmf(data_all[1:], 'zeroshot', 'percentile_error')
marginal_c3 = marginal_pmf(data_all[1:], 'model_type', 'percentile_error')

marginal_d1 = marginal_pmf(data_all[1:], 'ssim', 'percentile_kappa')
marginal_d2 = marginal_pmf(data_all[1:], 'zeroshot', 'percentile_kappa')
marginal_d3 = marginal_pmf(data_all[1:], 'model_type', 'percentile_kappa')
# def poly2(x, a, b, c):
#     return a * x**2 + b * x + c

def poly2(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def find_intersection(fun1, fun2, x0):
    return fsolve(lambda x : fun1(x) - fun2(x), x0)


#  Randomness

params_f1, _ = curve_fit(poly2, marginal_x1.index, marginal_x1.values)
params_f2, _ = curve_fit(poly2, marginal_y1.index, marginal_y1.values)
params_f3, _ = curve_fit(poly2, marginal_a1.index, marginal_a1.values)
params_f4, _ = curve_fit(poly2, marginal_b1.index, marginal_b1.values)
params_f5, _ = curve_fit(poly2, marginal_c1.index, marginal_c1.values)
params_f6, _ = curve_fit(poly2, marginal_d1.index, marginal_d1.values)


fitted_f1 = lambda x: poly2(x, *params_f1)
fitted_f2 = lambda x: poly2(x, *params_f2)
fitted_f3 = lambda x: poly2(x, *params_f3)
fitted_f4 = lambda x: poly2(x, *params_f4)
fitted_f5 = lambda x: poly2(x, *params_f5)
fitted_f6 = lambda x: poly2(x, *params_f6)

intersection_x = 0.976350
intersection_y1 = fitted_f1(intersection_x)
intersection_y2 = fitted_f2(intersection_x)

x_range = np.linspace(min(data_all[1:]['ssim']), max(data_all[1:]['ssim']), 100)

plt.figure(figsize=(10, 6))
plt.scatter(marginal_x1.index, marginal_x1.values, color='blue')
plt.scatter(marginal_y1.index, marginal_y1.values,color='red')
plt.scatter(marginal_a1.index, marginal_a1.values, color='orange')
plt.scatter(marginal_b1.index, marginal_b1.values,color='green')
plt.scatter(marginal_c1.index, marginal_c1.values, color='black')
plt.scatter(marginal_d1.index, marginal_d1.values,color='purple')
plt.plot(x_range, fitted_f1(x_range), label='Error Rate', color='blue', linestyle='--')
plt.plot(x_range, fitted_f2(x_range), label='Kappa', color='red', linestyle='--')
plt.plot(x_range, fitted_f3(x_range), label='Std Error', color='orange', linestyle='--')
plt.plot(x_range, fitted_f4(x_range), label='Std Kappa', color='green', linestyle='--')
plt.plot(x_range, fitted_f5(x_range), label='percentile_error', color='black', linestyle='--')
plt.plot(x_range, fitted_f6(x_range), label='percentile_kappa', color='purple', linestyle='--')

plt.axvline(x=intersection_x, color='cyan')
plt.plot(intersection_x, intersection_y1, '*', color='cyan', markersize=20)
plt.plot(intersection_x, intersection_y2, '*', color='cyan', markersize=20)
print(intersection_x, intersection_y1)
print(intersection_x, intersection_y2)
plt.xlabel('ssim',fontsize=18)
plt.ylabel('marginal distribution',fontsize=18)
# plt.title('Marginal Distributions and Fitted Curves for window size with trade-off point',fontsize=18)
# plt.legend()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=22)
plt.show()

#  zeroshot
data_all['zeroshot'][1:] = data_all['zeroshot'][1:] / (50 + data_all['zeroshot'][1:])

marginal_x1 = marginal_pmf(data_all[1:], 'ssim', 'error')
marginal_x2 = marginal_pmf(data_all[1:], 'zeroshot', 'error')
marginal_x3 = marginal_pmf(data_all[1:], 'model_type', 'error')

marginal_y1 = marginal_pmf(data_all[1:], 'ssim', 'kappa')
marginal_y2 = marginal_pmf(data_all[1:], 'zeroshot', 'kappa')
marginal_y3 = marginal_pmf(data_all[1:], 'model_type', 'kappa')

marginal_a1 = marginal_pmf(data_all[1:], 'ssim', 'Std Error')
marginal_a2 = marginal_pmf(data_all[1:], 'zeroshot', 'Std Error')
marginal_a3 = marginal_pmf(data_all[1:], 'model_type', 'Std Error')

marginal_b1 = marginal_pmf(data_all[1:], 'ssim', 'Std Kappa')
marginal_b2 = marginal_pmf(data_all[1:], 'zeroshot', 'Std Kappa')
marginal_b3 = marginal_pmf(data_all[1:], 'model_type', 'Std Kappa')

marginal_c1 = marginal_pmf(data_all[1:], 'ssim', 'percentile_error')
marginal_c2 = marginal_pmf(data_all[1:], 'zeroshot', 'percentile_error')
marginal_c3 = marginal_pmf(data_all[1:], 'model_type', 'percentile_error')

marginal_d1 = marginal_pmf(data_all[1:], 'ssim', 'percentile_kappa')
marginal_d2 = marginal_pmf(data_all[1:], 'zeroshot', 'percentile_kappa')
marginal_d3 = marginal_pmf(data_all[1:], 'model_type', 'percentile_kappa')

params_f1, _ = curve_fit(poly2, marginal_x2.index, marginal_x2.values)
params_f2, _ = curve_fit(poly2, marginal_y2.index, marginal_y2.values)
params_f3, _ = curve_fit(poly2, marginal_a2.index, marginal_a2.values)
params_f4, _ = curve_fit(poly2, marginal_b2.index, marginal_b2.values)
params_f5, _ = curve_fit(poly2, marginal_c2.index, marginal_c2.values)
params_f6, _ = curve_fit(poly2, marginal_d2.index, marginal_d2.values)


fitted_f1 = lambda x: poly2(x, *params_f1)
fitted_f2 = lambda x: poly2(x, *params_f2)
fitted_f3 = lambda x: poly2(x, *params_f3)
fitted_f4 = lambda x: poly2(x, *params_f4)
fitted_f5 = lambda x: poly2(x, *params_f5)
fitted_f6 = lambda x: poly2(x, *params_f6)

intersection_x = 0.166999
intersection_y1 = fitted_f1(intersection_x)
intersection_y2 = fitted_f2(intersection_x)


x_range = np.linspace(min(data_all[1:]['zeroshot']), max(data_all[1:]['zeroshot']), 100)
# x_range = np.linspace(10, 100)

plt.figure(figsize=(10, 6))
marginal_x2.index = [0,0.167,0.286,0.375,0.444,0.500]
marginal_y2.index = [0,0.167,0.286,0.375,0.444,0.500]
plt.scatter(marginal_x2.index, marginal_x2.values, color='blue')
plt.scatter(marginal_y2.index, marginal_y2.values,color='red')
plt.scatter(marginal_a2.index, marginal_a2.values, color='orange')
plt.scatter(marginal_b2.index, marginal_b2.values,color='green')
plt.scatter(marginal_c2.index, marginal_c2.values, color='black')
plt.scatter(marginal_d2.index, marginal_d2.values,color='purple')
plt.plot(x_range, fitted_f1(x_range), label='Error Rate', color='blue', linestyle='--')
plt.plot(x_range, fitted_f2(x_range), label='Kappa', color='red', linestyle='--')
plt.plot(x_range, fitted_f3(x_range), label='Std Error', color='orange', linestyle='--')
plt.plot(x_range, fitted_f4(x_range), label='Std Kappa', color='green', linestyle='--')
plt.plot(x_range, fitted_f5(x_range), label='percentile_error', color='black', linestyle='--')
plt.plot(x_range, fitted_f6(x_range), label='percentile_kappa', color='purple', linestyle='--')


# plt.scatter(intersection_x, intersection_y, color='green', marker='o')
plt.axvline(x=intersection_x, color='cyan')
plt.plot(intersection_x, intersection_y1, '*', color='cyan', markersize=20)
plt.plot(intersection_x, intersection_y2, '*', color='cyan', markersize=20)
# plt.text(intersection_x, 0, f'{intersection_x[0]:.2f}', horizontalalignment='center', verticalalignment='bottom', color='green')

plt.xlabel('zeroshot',fontsize=18)
plt.ylabel('marginal distribution',fontsize=18)
# plt.title('Marginal Distributions and Fitted Curves for zeroshot with trade-off point',fontsize=18)
print(intersection_x, intersection_y1)
print(intersection_x, intersection_y2)
# plt.legend()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='upper right',fontsize=22)
plt.show()

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



amount_list = [5.3e6, 7.8e6, 9.2e6, 12e6, 19e6, 30e6, 43e6, 66e6]
log_x1 = np.log(amount_list)

params_f1, _ = curve_fit(poly2, log_x1, marginal_x3.values)
params_f2, _ = curve_fit(poly2, log_x1, marginal_y3.values)
params_f3, _ = curve_fit(poly2, log_x1, marginal_a3.values)
params_f4, _ = curve_fit(poly2, log_x1, marginal_b3.values)
params_f5, _ = curve_fit(poly2, log_x1, marginal_c3.values)
params_f6, _ = curve_fit(poly2, log_x1, marginal_d3.values)

fitted_f1 = lambda x: poly2(x, *params_f1)
fitted_f2 = lambda x: poly2(x, *params_f2)
fitted_f3 = lambda x: poly2(x, *params_f3)
fitted_f4 = lambda x: poly2(x, *params_f4)
fitted_f5 = lambda x: poly2(x, *params_f5)
fitted_f6 = lambda x: poly2(x, *params_f6)

x_range = np.linspace(min(log_x1), max(log_x1), 1000)

intersection_x = np.log(43e6)
intersection_y1 = fitted_f1(intersection_x)
intersection_y2 = fitted_f2(intersection_x)
print(intersection_x, intersection_y1)
print(intersection_x, intersection_y2)


available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16','RN50x64']

# Plot the data, the fitted curves, and the intersection point
plt.figure(figsize=(10, 6))
plt.scatter(log_x1, marginal_x3.values, color='blue')
plt.scatter(log_x1, marginal_y3.values, color='red')
plt.scatter(log_x1, marginal_a3.values, color='orange')
plt.scatter(log_x1, marginal_b3.values, color='green')
plt.scatter(log_x1, marginal_c3.values, color='black')
plt.scatter(log_x1, marginal_d3.values, color='purple')
plt.plot(x_range, [fitted_f1(x) for x in x_range], label='Error Rate', color='blue', linestyle='--')
plt.plot(x_range, [fitted_f2(x) for x in x_range], label='Kappa', color='red', linestyle='--')
plt.plot(x_range, [fitted_f3(x) for x in x_range], label='Std Error Rate', color='orange', linestyle='--')
plt.plot(x_range, [fitted_f4(x) for x in x_range], label='Std Kappa', color='green', linestyle='--')
plt.plot(x_range, [fitted_f5(x) for x in x_range], label='percentile_error', color='black', linestyle='--')
plt.plot(x_range, [fitted_f6(x) for x in x_range], label='percentile_kappa', color='purple', linestyle='--')
# plt.scatter(intersection_x, intersection_y, color='green', marker='o')
plt.axvline(x=intersection_x, color='cyan')
plt.plot(intersection_x, intersection_y1, '*', color='cyan', markersize=20)
plt.plot(intersection_x, intersection_y2, '*', color='cyan', markersize=20)
# plt.text(intersection_x, min(plt.ylim()), f'{intersection_x:.2f}', horizontalalignment='center', verticalalignment='bottom', color='green')
# plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xticks(log_x1, [f"{int(x/1e6)}M" for x in amount_list])
plt.xlabel('number of parameters',fontsize=18)
plt.ylabel('marginal distribution',fontsize=18)
# plt.title('Marginal Distributions and Fitted Curves for number of parameters with trade-off point',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=20)
plt.legend(fontsize=22)
plt.show()
