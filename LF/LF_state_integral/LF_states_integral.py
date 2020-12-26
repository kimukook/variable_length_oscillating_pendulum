import numpy as np
import matplotlib.pyplot as plt
import pendulum
from pendulum import Pendulum
from functools import partial
from scipy import io
'''
This is a script that computes a specific function over the state space (phase space). The function considered here
is for proving the asymptotic stability for the variable-length pendulum problem. 

=====================================
Author  :  Muhan Zhao
Date    :  Dec. 11, 2019
Location:  UC San Diego, La Jolla, CA
=====================================
'''


class FTLF:
    def __init__(self, attributes):
        self.d = attributes.get('finite-time d', 0.2)
        self.dt = None

    @staticmethod
    def compute_lf(self, vary_length_pendulum: Pendulum):
        self.dt = vary_length_pendulum.dt
        steps = round(self.d / self.dt)
        if steps > vary_length_pendulum.steps:
            raise ValueError('FTLF time length > pendulum simulated time steps! Reduce d.')
        else:
            pass
        assemble_states = np.vstack((vary_length_pendulum.asym_control_phi, vary_length_pendulum.asym_control_dphi))
        w = np.sum(np.linalg.norm(assemble_states, axis=0))
        return w


def execute_pendulum_control(wave, attributes):
    vary_length_pendulum = Pendulum(wave, attributes)
    vary_length_pendulum.main()
    return vary_length_pendulum


def compute_lyapunov_function_values(vary_length_pendulum, d):
    attributes_ftlf = {
        'finite-time d': d
    }
    # define the Finite time Lyapunov Function class
    ftlf = FTLF(attributes_ftlf)

    # compute the Lyapunov Function value at each point
    return ftlf.compute_lf(ftlf, vary_length_pendulum)


# Define the attributes for the variable-length pendulum problem
d = .2
T = .6
dt = 0.001
g = 9.8
l0 = 1
m = 1

attributes = {
    'm': m,
    'max_t': T,
    'dt': dt,
    'constrain_L': False,
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
    'g': 1
}

# design the discretization of the phase space
size = 100
width = 3
# define the region of interest
x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-width, width, size))
W = np.zeros(x.shape)

for i in range(x.shape[0]):
    print(f'i = {i}')
    for j in range(y.shape[1]):
        phi = x[i, j] * np.ones(1)
        dphi = y[i, j] * np.ones(1)

        wave = {
            'phi': phi,
            'dphi': dphi,
        }

        vary_length_pendulum = execute_pendulum_control(wave, attributes)

        # find the LF value of each point
        W[i, j] = compute_lyapunov_function_values(vary_length_pendulum, d)

LF = {
    'width': width,
    'size': size,
    'LF': W
}
io.savemat('LF_states_integral.mat', LF)


# plot
fig = plt.figure(figsize=[12, 8])
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
mu = plt.contour(x, y, W, levels=20, zorder=-1)
cbar = plt.colorbar(mu)

plt.xlabel(r'$\phi(t)$', size=20)
plt.ylabel(r'$\dot{\phi}(t)$', size=20, rotation=0)

plt.ylim(-2, 2)
plt.xlim(-3, 3)
plt.show()
# plt.savefig('LF_StateIntegral.png', format='png', dpi=300)
plt.close(fig)
