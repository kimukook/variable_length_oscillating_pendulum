import numpy as np
import matplotlib.pyplot as plt
import pendulum
from pendulum import Pendulum
from scipy import io

'''
This is a script that generates the vector field of the dynamic of closed-loop control of variable-length pendulu.
After the discretization of the phase space, compute the next state carried by the system dynamic applied with the 
proposed feedback rule for each single point in the phase space. Finally, generate the plot based on the time-change of
states.

In this script, the time change, delta = .2

Result: 
1. Asymptotic stability towards the orgin;
2. As the system is close to the origin, the magnitude of the change becomes smaller.
3. From this plot it is not cleat to see that all the trajectories are driving towards the origin.
So the thinking is that, increase the time-length of the system with point in the phase space.  

=====================================
Author  :  Muhan Zhao
Date    :  Dec. 20, 2019
Location:  UC San Diego, La Jolla, CA
=====================================
'''


def execute_pendulum_control(wave, attributes):
    vary_length_pendulum = Pendulum(wave, attributes)
    vary_length_pendulum.main()
    return vary_length_pendulum


def plot_finite_time_trajectory(vary_length_pendulum):
    # plot the trajectory with the time length=0.2
    plt.plot(vary_length_pendulum.asym_control_phi, vary_length_pendulum.asym_control_dphi, 'k')

    # plot the arrow on the line, to indicate the moving direction of the trajectory
    arrow_start = int(round(vary_length_pendulum.steps / 2))
    plt.arrow(vary_length_pendulum.asym_control_phi[arrow_start], vary_length_pendulum.asym_control_dphi[arrow_start],
              vary_length_pendulum.asym_control_phi[arrow_start + 1] - vary_length_pendulum.asym_control_phi[
                  arrow_start],
              vary_length_pendulum.asym_control_dphi[arrow_start + 1] - vary_length_pendulum.asym_control_dphi[
                  arrow_start]
              , color='k', shape='full', lw=0, length_includes_head=True, head_width=.1)


# define the plot size for the vector field
size = 10
width = 1
# define the region of interest
x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-width, width, size))


# wrap the input for the class of pendulum
d = .2
dt = 0.001
g = 9.8
l0 = 1
m = 1

attributes = {
    'm': m,
    'max_t': d,
    'dt': dt,
    'adaptive_mode': False,
    'delta_adaptive_const': .15,
    'asymptotic_mode': True,
    'delta_asymptotic_const': .05,
    'l0': l0,
    'constrain_L': False,
    'Ldotmax': 5,
    'Ldotmin': -5,
    'Lmax': 1.5,
    'Lmin': 0.5,
    'g': g,
    'plot': False,
    'save_fig': False,
    'show_fig': False,
    'save_data': False,
}


# Find the trajectory (time window = 0.2, the same as finite-time d) for each point defined in the phase space

fig = plt.figure(figsize=[12, 8])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()

for i in range(x.shape[0]):
    print(f'i = {i}')
    for j in range(y.shape[1]):
        phi = x[i, j] * np.ones(1)
        dphi = y[i, j] * np.ones(1)

        wave = {
            'phi': phi,
            'dphi': dphi,
        }

        # assemble pendulum class
        vary_length_pendulum = execute_pendulum_control(wave, attributes)

        # plot the trajectory for each pair of initial condition in phase space
        plot_finite_time_trajectory(vary_length_pendulum)


plt.xlabel(r'$\phi(t)$', size=20)
plt.ylabel(r'$\dot{\phi}(t)$', size=20, rotation=0)
# # define the plot size for the vector field


# read input W from different setting of LFs:
# data = io.loadmat('LF_states_integral.mat')
# W = data['LF']
# size = data['size']
# width = data['width']
# x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-width, width, size))

# mu = plt.contour(x, y, W, levels=20, zorder=-1)
# cbar = plt.colorbar(mu)

plt.ylim(-2, 2)
plt.xlim(-2, 2)
# plt.show()
# plt.savefig('Long_vector_field_in_StateIntegralLF.png', format='png', dpi=300)
plt.show()
plt.close(fig)

