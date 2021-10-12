import numpy as np
import matplotlib.pyplot as plt
from pendulum import Pendulum
from scipy import io

'''
This is a script that computes the region of attraction for proving the asymptotic stability 
of the Variable-Length Pendulum (VLP) problem. 

=====================================
Author  :  Muhan Zhao
Date    :  Jul. 23, 2021
Location:  UC San Diego, La Jolla, CA
=====================================
'''

# 1. Set up the starting points on the grid, computes the trajectory generated by VLP.
#    To see if the


def execute_pendulum_control(wave, attributes):
    vary_length_pendulum = Pendulum(wave, attributes)
    vary_length_pendulum.main()
    return vary_length_pendulum


def asymptotic_stable_judger(vary_length_pendulum):
    threshold = 1e-10
    signal = np.vstack((vary_length_pendulum.asym_control_phi, vary_length_pendulum.asym_control_dphi))
    left_x_crossers = np.empty(shape=[2, 0])

    # Step 1: find the points close to the y-axis from x left plane
    # Since the time step has been set to be .02, re-set the threshold to be 0.1
    while threshold <= 1e-1:
        left_x_crossers = signal[:, (abs(signal[1, :]) < threshold) & (signal[0, :] < 0)]
        # y-axis too aggressive, set the threshold larger
        if left_x_crossers.shape[1] <= 5:
            threshold *= 10
        else:
            # print('already found enough points near y axis, break')
            break
        # print('threshold reaches bound, no enough points near y axis found')

    # Step 2: clean the data points
    # criterion 1: though trajectory shrinking, for each slice there might be multiple points considered due to the
    # numerical issue
    if left_x_crossers.shape[1] <= 1:
        # only one point close to x-axis, not converging.
        roa = False
        slow_converge_rate = False

    else:
        delete_list = []
        query_point = left_x_crossers[0, 0]
        for i in range(1, left_x_crossers.shape[1]):
            if abs(query_point - left_x_crossers[0, i]) * 100 < 1:
                delete_list.append(i)
            else:
                query_point = left_x_crossers[0, i]
        left_x_crossers_cleaned = np.delete(left_x_crossers, delete_list, 1)
        if left_x_crossers_cleaned.shape[1] <= 1:
            roa = False
            slow_converge_rate = True
            # trajectory passing through x-axis through the same position, not converging/converging very slow due to
            # the numerical issue

        else:
            # condition 1: consider the x-value of the point crossing the x-axis at the left plane, if value increasing,
            # the trajectory is converging
            decrease_indicator = np.sign(np.diff(left_x_crossers_cleaned[0, :]))
            last_first_diff = left_x_crossers[0, -1] * 10 - left_x_crossers[0, 1]

            # if indicator all positive, inside ROA true
            if (decrease_indicator > 0).all() or last_first_diff > 0:
                roa = True
                slow_converge_rate = False

            else:
                roa = False
                slow_converge_rate = False

    return roa, slow_converge_rate


iter_max = 5

# Define the attributes for the variable-length pendulum problem
d = 10
dt = 0.02
g = 9.8
l0 = 1
m = 1
delta = .05

attributes = {
    'm': m,
    'max_t': d,
    'dt': dt,
    'constrain_L': True,
    'save_data': False,
    'plot': False,
    'save_fig': False,
    'show_fig': True,
    'asymptotic_mode': True,
    'delta_asymptotic_const': delta,
    'adaptive_mode': False,
    'delta_adaptive_const': delta,
    'l0': l0,
    'Ldotmax': 5,
    'Ldotmin': -5,
    'Lmax': 1.5,
    'Lmin': 0.5,
    'g': g,
}


size = 20
width = 2.5
# define the region of interest
x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-width, width, size))

roa_indicator = np.zeros(x.shape)
converge_slow_indicator = np.zeros(x.shape)


for i in range(x.shape[0]):

    for j in range(y.shape[1]):
        print(f'i = {i} || j = {j}')
        phi = x[i, j] * np.ones(1)
        dphi = y[i, j] * np.ones(1)

        wave = {
            'phi': phi,
            'dphi': dphi,
        }

        vary_length_pendulum = execute_pendulum_control(wave, attributes)
        roa_indicator[i, j], converge_slow_indicator[i, j] = asymptotic_stable_judger(vary_length_pendulum)


# fig = plt.figure(figsize=[9, 9])
# plt.grid()
# mu = plt.contour(x, y, roa_indicator, levels=20)
# cbar = plt.colorbar(mu)
#
# plt.xlabel(r'$\phi(t)$', size=20)
# plt.ylabel(r'$\dot{\phi}(t)$', size=20, rotation=0)
#
# plt.ylim(-2, 2)
# plt.xlim(-2, 2)
# plt.show()
# plt.savefig('true_roa_VLP.png', format='png', dpi=300)
# plt.close(fig)

# fig = plt.figure(figsize=[9, 9])
# plt.grid()
# plt.scatter(x, y, 'b')
#
# list_indicator = np.where(roa_indicator > 0)

