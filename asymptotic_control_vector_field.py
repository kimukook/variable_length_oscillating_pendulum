import numpy as np
import matplotlib.pyplot as plt
import pendulum
from pendulum import Pendulum
from functools import partial
from scipy import io

'''
This is a script that generates the vector field of the dynamic of closed-loop control of variable-length pendulu.
After the discretization of the phase space, compute the next state carried by the system dynamic applied with the 
proposed feedback rule for each single point in the phase space. Finally, generate the plot based on the time-change of
states.

In this script, the time change, delta = .001

Result: 
1. Asymptotic stability towards the orgin;
2. As the system is close to the origin, the magnitude of the change becomes smaller.
3. From this plot it is not cleat to see that all the trajectories are driving towards the origin.
So the thinking is that, increase the time-length of the system with point in the phase space.  

=====================================
Author  :  Muhan Zhao
Date    :  Dec. 11, 2019
Location:  UC San Diego, La Jolla, CA
=====================================
'''


def execute_pendulum_control(wave, attributes):
    vary_length_pendulum = Pendulum(wave, attributes)
    vary_length_pendulum.main()
    return vary_length_pendulum


def compute_vector_field(vary_length_pendulum, states):
    new_states = np.hstack((vary_length_pendulum.asym_control_phi[0], vary_length_pendulum.asym_control_dphi[0]))
    # if np.linalg.norm(new_states-states) > 3:
    #     print('ss')
    u_sub, v_sub = np.copy(new_states - states)
    return u_sub, v_sub


# define the plot size for the vector field
size = 10
width = 2
# define the region of interest
# x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-4, 3, size))
x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-width, width, size))
# define the field vector (u, v) of each point
u, v = np.zeros(x.shape), np.zeros(x.shape)

# wrap the input for the class of pendulum

T = .1
dt = 0.02
g = 9.8
m = 1
l = 1

properties = {
    'm': m,
    'max_t': T,
    'dt': dt,
    'adaptive_mode': False,
    'delta_adaptive_const': .15,
    'asymptotic_mode': True,
    'delta_asymptotic_const': .05,
    'l0': l,
    'control_L': True,
    'Ldotmax': 5,
    'Ldotmin': -5,
    'Lmax': 1.5,
    'Lmin': 0.5,
    'g': g,
    'plot': False,
    'save_fig': False,
    'show_fig': False,
    'save_data': False
}


# Find the vector field for each point defined in the domain


for i in range(x.shape[0]):
    print(f'i = {i}')
    for j in range(y.shape[1]):
        phi = x[i, j] * np.ones(1)
        dphi = y[i, j] * np.ones(1)
        # if i == 2 and j == 14:
        #     print('ss') werid arrow
        wave = {
            'phi': phi,
            'dphi': dphi,
        }

        states = np.hstack((phi, dphi))

        # assemble pendulum class
        vary_length_pendulum = execute_pendulum_control(wave, properties)

        # find the vector field direction of each point
        u[i, j], v[i, j] = compute_vector_field(vary_length_pendulum, states)


fig = plt.figure(figsize=[16, 16])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.quiver(x, y, u, v, zorder=0)
plt.xlabel(r'$\phi(t)$', size=20)
plt.ylabel(r'$\dot{\phi}(t)$', rotation=0, size=20, labelpad=20)

data = io.loadmat('pendulum_data.mat')
phi = data['asym_phi']
dphi = data['asym_dphi']
plt.plot(phi[0], dphi[0], 'r--', label='Asymptotic', zorder=1)

plt.savefig('vector_field.png', format='png', dpi=300)

plt.show()
plt.close(fig)
