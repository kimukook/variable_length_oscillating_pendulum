import numpy as np
import matplotlib.pyplot as plt
import pendulum
from pendulum import Pendulum
from finite_time_LF import FTLF

from functools import partial

'''
This is a script that generates the vector field of the dynamic of closed-loop control of variable-length pendulu.

=====================================
Author  :  Muhan Zhao
Date    :  Dec. 11, 2019
Location:  UC San Diego, La Jolla, CA
=====================================
'''


def set_ddphi(phi, dphi):
    if dphi >= 0 > phi:
        ddphi = - (dphi**2/phi) * 2

    elif dphi <= 0 < phi:
        ddphi = 2 * (-dphi**2/phi)

    elif dphi > 0 and phi >= 0:
        ddphi = (-dphi**2/phi)/2

    elif dphi < 0 and phi <= 0:
        ddphi = (-dphi**2/phi) / 2
    else:
        ddphi = 2 * np.ones(1)
    return ddphi


def execute_pendulum_control(wave, attributes):
    vary_length_pendulum = Pendulum(wave, attributes)
    vary_length_pendulum.main()

    return vary_length_pendulum


def compute_vector_field(vary_length_pendulum, states):
    new_states = np.hstack((vary_length_pendulum.asym_control_phi[0], vary_length_pendulum.asym_control_dphi[0]))
    u_sub, v_sub = np.copy(new_states - states)
    return u_sub, v_sub


def compute_lyapunov_function_values(vary_length_pendulum, d):
    attributes_ftlf = {
        'finite-time d': d
    }
    # define the Finite time Lyapunov Function class
    ftlf = FTLF(attributes_ftlf)

    # compute the Lyapunov Function value at each point
    return ftlf.compute_lf(ftlf, vary_length_pendulum)


# define the plot size for the vector field
size = 10
width = 2
# define the region of interest
x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-width, width, size))
# define the field vector (u, v) of each point
u, v, W = np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape)

# wrap the input for the class of pendulum
a = 2
d = .2
T = .6
dt = 0.001
g = 9.8
l = 1
w0 = np.sqrt(g / l)

attributes = {
    'max_t': T,
    'dt': dt,
    'plot': False,
    'save_fig': False,
    'show_fig': False,
    'adaptive_mode': False,
    'delta_adaptive_const': .15,
    'asymptotic_mode': True,
    'delta_asymptotic_const': .1,
    'l0': 1,
    'Ldotmax': 5,
    'Ldotmin': -5,
    'Lmax': 1.5,
    'Lmin': 0.5,
    'g': 1

}


# Find the vector field for each point defined in the domain


for i in range(x.shape[0]):
    for j in range(y.shape[1]):
        phi = x[i, j] * np.ones(1)
        dphi = y[i, j] * np.ones(1)
        ddphi = set_ddphi(phi, dphi)

        wave = {
            'amplitude': a,
            'frequency': w0,
            'phi': phi,
            'dphi': dphi,
            'ddphi': ddphi
        }

        states = np.hstack((phi, dphi))
        # assemble pendulum class
        vary_length_pendulum = execute_pendulum_control(wave, attributes)

        # find the vector field direction of each point
        u[i, j], v[i, j] = compute_vector_field(vary_length_pendulum, states)

        # find the LF value of each point
        W[i, j] = compute_lyapunov_function_values(vary_length_pendulum, d)


fig = plt.figure(figsize=[9, 9])
plt.grid()
plt.quiver(x, y, u, v)
plt.xlabel(r'$\phi(t)$')
plt.ylabel(r'$\dot{\phi}(t)$')
plt.savefig('vector_field.png', format='png', dpi=300)
# plt.show()
plt.close(fig)

# add colorbar
fig = plt.figure()
mu = plt.contour(x, y, W)
cbar = plt.colorbar(mu)
plt.grid()
# plt.show()
plt.xlabel(r'$\phi(t)$')
plt.ylabel(r'$\dot{\phi}(t)$')
plt.savefig('LF_contour.png', format='png', dpi=300)

plt.close(fig)

