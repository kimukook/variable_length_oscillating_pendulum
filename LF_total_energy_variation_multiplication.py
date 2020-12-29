import numpy as np
import matplotlib.pyplot as plt
import pendulum
from pendulum import Pendulum
from functools import partial
from scipy import io
'''
This is a script that computes a specific function over the state space (phase space). The function considered here
is for proving the asymptotic stability for the variable-length pendulum problem. 

V * L
=====================================
Author  :  Muhan Zhao
Date    :  Dec. 26, 2019
Location:  UC San Diego, La Jolla, CA
=====================================
'''


def compute_total_energy_LF(p: Pendulum):
    '''
    Compute the total energy of the variable-length pendulum,
    V = 1/2 * m (L^2 * [d(phi)/dt]^2 + Ldot ^2) + m * g * (L0 - L * cos phi)
    :param p:
    :return:
    '''
    # assemble the staets
    state = np.hstack((p.wave_phi[-1], p.wave_dphi[-1]))

    # compute the length and time-change of the length
    L = p.compute_length(state)
    Ldot = p.compute_length_dot(state)

    # Compute the total energy
    V = 1/2 * p.m * (L**3 * state[1]**2 + Ldot**2 * L) + p.m * p.g * (p.l0 * L - L**2 * np.cos(state[0]))
    # # The code below is for
    # if V >= 50:
    #     V = 50
    return V


def compute_total_energy_LF_derivative(p: Pendulum):
    phi = np.hstack((p.wave_phi[-1], p.wave_dphi[-1]))
    f = p.variable_length_eom(phi)

    dVdphi = 1/2 * p.m * (p.l0**3*3*(1+p.delta*phi[0]*phi[1])**2*p.delta*phi[1]**3+p.l0**3*p.delta**3*phi[1]*(phi[0]*f[1]+phi[1]**2)**2+p.l0**3*p.delta**2*2*(1+p.delta*phi[0]*phi[1])*(phi[0]*f[1]+phi[1]**2)*f[1])\
             +p.g*p.m*(p.l0**2*p.delta*phi[1]-2*p.l0**2*(1+p.delta*phi[0]*phi[1])*p.delta*phi[1]*np.cos(phi[0])+np.sin(phi[0])*p.l0**2*(1+p.delta*phi[0]*phi[1])**2)

    dVddphi = 1/2 * p.m * (p.l0**3*3*(1+p.delta*phi[0]*phi[1])**2*p.delta*phi[0]*phi[1]**2+p.l0**3*(1+p.delta*phi[0]*phi[1])**3*2*phi[1]+p.l0**3*p.delta**3*phi[0]*(phi[0]*f[1]+phi[1]**2)**2+p.l0**3*(1+p.delta*phi[0]*phi[1])*p.delta**2*2*(phi[0]*f[1]+phi[1]**2)*2*phi[1])\
              +p.g*p.m*(p.l0**2*p.delta*phi[0]-p.l0**2*np.cos(phi[0])*2*(1+p.delta*phi[0]*phi[1])*p.delta*phi[0])
    dV = dVdphi * phi[1] + dVddphi * f[1]

    if abs(dV) < 1e-4:
        return 0
    else:
        if dV < -1e-4:
            return -1
        elif dV > 1e-4:
            return 1
        else:
            pass


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


# design the discretization of the phase space
size = 1000
width = 2
# define the region of interest
x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-width, width, size))
# x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-4, 3, size))
W, dW = np.zeros(x.shape), np.zeros(x.shape)

for i in range(x.shape[0]):
    print(f'i = {i}')
    for j in range(y.shape[1]):
        phi = x[i, j] * np.ones(1)
        dphi = y[i, j] * np.ones(1)

        wave = {
            'phi': phi,
            'dphi': dphi,
        }

        pending_pendulum = Pendulum(wave, attributes)
        # find the LF value of each point
        W[i, j] = compute_total_energy_LF(pending_pendulum)
        dW[i, j] = compute_total_energy_LF_derivative(pending_pendulum)

LF = {
    'width': width,
    'size': size,
    'LF': W,
    'LFdot': dW
}
io.savemat('LF_total_energy_variation_multiplication.mat', LF)


# plot
fig = plt.figure(figsize=[9, 9])
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
mu = plt.contour(x, y, W, levels=30, colors='gray')
cbar = plt.colorbar(mu)

plt.xlabel(r'$\phi(t)$', size=20)
plt.ylabel(r'$\dot{\phi}(t)$', size=20, rotation=0)

plt.contourf(x, y, dW, cmap='Reds', alpha=.3)

# data = io.loadmat('pendulum_data.mat')
# phi = data['asym_phi']
# dphi = data['asym_dphi']
# plt.plot(phi[0], dphi[0], 'b--', label='Asymptotic', zorder=1)

# plt.show()
plt.savefig('LF_TotalEnergy_Vdot_multiplication.png', format='png', dpi=300)
plt.close(fig)


# # read a specific trajectory
