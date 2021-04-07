import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import matplotlib.tri as tri
import pendulum
import gurobipy as grp
from gurobipy import GRB
from scipy.spatial import Delaunay

'''
Reference: Sigurður Hafstein et al. Continuous and Piecewise Affine Lyapunov Functions using the Yoshizawa Construction

=====================================
Author  :  Muhan Zhao
Date    :  Jan. 15, 2020
Location:  UC San Diego, La Jolla, CA
=====================================
'''

# 1. Generate the Cartesian grid and the Triangulation
width = 1.5
size = 10

xPoints, yPoints = np.linspace(-width, width, size+1), np.linspace(-width, width, size+1)
gridPoints = np.array([[[x, y] for y in yPoints] for x in xPoints])
# tri_simplices = []
# tri_simplices += [[i + j * (size + 1), (i+1)+j*(size+1), i+(j+1)*(size+1)] for i in range(size) for j in range(size)]
# tri_simplices += [[(size-i-1)+(size-j)*(size+1), (size-i)+(size-j)*(size+1), (size-i)+(size-j-1)*(size+1)] for i in range(size) for j in range(size)]
#
# triang = tri.Triangulation(gridPoints[:, :, 0].flatten(), gridPoints[:, :, 1].flatten(), tri_simplices)
#
# fig = plt.figure(figsize=[9, 9])
# plt.triplot(triang)
# plt.plot(gridPoints[:, :, 0], gridPoints[:, :, 1], 'ko')
# # plt.show()
# plt.savefig('triangulation.png', format='png', dpi=300)
# plt.close(fig)


grids = np.vstack((gridPoints[:, :, 0].flatten(), gridPoints[:, :, 1].flatten()))
# 2. Find beta(s,t) that satisfy eqn (4) in CPA-Yoshizawa

# set pendulum parameters
d = .2
T = 1
dt = 0.005
g = 9.8
l0 = 1
m = 1

attributes = {
    'm': m,
    'max_t': T,
    'dt': dt,
    'constrain_L': True,
    'save_data': False,
    'plot': False,
    'save_fig': False,
    'show_fig': False,
    'asymptotic_mode': True,
    'delta_asymptotic_const': .1,
    'adaptive_mode': False,
    'delta_adaptive_const': .05,
    'l0': l0,
    'Ldotmax': 5,
    'Ldotmin': -5,
    'Lmax': 1.5,
    'Lmin': 0.5,
    'g': g
}
# compute the dynamic trajectories for all grid points (
trajectories = np.zeros((grids.shape[1], int(T/dt)*2))
steps = int(T/dt)

# for ind, point in enumerate(grids.T):
#     print(f'ind = {ind}')
    # print(f'ind = {ind}')
    # print(f'point = {point}')
    # if ind >= 2:
    #     break
ind = 0
point = grids.T[0]
wave = {
    'phi': point[0],
    'dphi': point[1]
}
vary_length_pendulum = pendulum.Pendulum(wave, attributes)
vary_length_pendulum.main()
states = np.hstack((vary_length_pendulum.asym_control_phi, vary_length_pendulum.asym_control_dphi))
trajectories[ind, :] = np.copy(states)


t = np.arange(0, T, dt)
x0 = grids.T[0]
beta = np.linalg.norm(x0)*np.exp(-.05*t)
states = np.vstack((trajectories[0, :steps], trajectories[1, :steps]))
phi = np.linalg.norm(states, axis=0)

fig = plt.figure(figsize=[16, 9])
plt.plot(t, beta, 'r--')
plt.plot(t, phi, 'k')
plt.show()
