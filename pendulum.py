import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy import io

'''
This is a script that simulates the movement of variable-length pendulum.
equation of motion:

   ..           .    .
  theta = [-2 * L * phi - g * sin(theta)] / l

The stabilization of this variable length pendulum is realized via the simple feedback control of the pendulum length. 
The feedback control rule has the following form:

                             .
L = L0 * (1 + delta * phi * phi)

References lead to: 
Bewley(2020) Stabilization of low-altitude balloon systems, Part 1: rigging with a single taut ground tether, 
with analysis as a variable-length pendulum

=====================================
Author  :  Muhan Zhao
Date    :  Oct. 16, 2019
Location:  UC San Diego, La Jolla, CA
=====================================
'''


class Pendulum:
    """
    Oscillating Pendulum Simulator: including a novel feedback rule to stabilize the pendulum regardless the
    parameters {m, g, L0} and the exact expression of the oscillating angle.

    Parameters
    ------------

    :param wave       : dict, including parameters:
    :keys frequency   : float, the frequency of the oscillating angle
    :keys phi         : array, the valeus of oscillating angle w.r.t. time
    :keys dphi        : array, the valeus of oscillating angle's 1st-order derivative w.r.t. time
    :keys ddphi       : array, the valeus of oscillating angle's 2nd-order derivative w.r.t. time

    :param attributes            : dict, including parameters:
    :keys max_t                  : float, the maximum time length, default 30
    :keys dt                     : float, the marching time step, default 0.001
    :keys delta_mode_adaptive    : string, adaptive mode is on or no
    :keys delta_adaptive_const   : float, the parameter C for adaptive delta, default 0.15
    :keys delta_mode_asymptotic  : string, asymptotic mode is on or no
    :keys delta_asymptotic_const : float, the value of delta for feedback rule, default 0.1
    :keys l0                     : float, initial length of pendulum, default 1
    :keys Lmax                   : float, maximum length of pendulum, default l0*1.2
    :keys Lmin                   : float, minimum length of pendulum, default l0*.8
    :keys Ldotmax                : float, maximum derivative of pendulum length, default 1.2
    :keys Ldotmin                : float, minimum derivative of pendulum length, default .8
    :keys save_fig               : bool, save fig or no
    :keys show_fig               : bool, display fig or no
    :keys format_fig             : string, fig format to be saved

    ------------

    Methods
    ------------

    :method free_pendulum_oscillation
    :method control_pendulum_oscillation
    :method adaptive_control_pendulum_oscillation
    ------------
    """
    def __init__(self, wave, attributes):
        self.max_t = attributes.get('max_t', 30)
        self.dt = attributes.get('dt', 0.001)

        # steps: the time slides of control inputs
        # Problem here: sometimes, even the result of max_t/dt is an integer, but it could be
        # not exactly the input you want, because how python stores the number. Sometimes there will be
        # a small error such like 1e-10. So thats why need round here.
        self.steps = int(round(self.max_t / self.dt))
        self.time = np.linspace(0, self.max_t, self.steps)

        self.l0 = attributes.get('l0', 1)
        self.Ldotmax = attributes.get('Ldotmax', 5)
        self.Ldotmin = attributes.get('Ldotmin', -5)

        self.Lmax = attributes.get('Lmax', 1.2 * self.l0)
        self.Lmin = attributes.get('Lmin', .8 * self.l0)

        self.g = attributes.get('g', 9.8)
        self.m = attributes.get('m', 1)

        self.wave_phi = wave['phi']
        self.wave_dphi = wave['dphi']

        self.frequency = wave.get('frequency', np.pi/2)
        self.control_start_time = self.dt * self.wave_phi.shape[0]
        self.prev_length = self.wave_phi.shape[0]
        self.entire_t = np.arange(0, self.dt * self.prev_length + self.max_t, self.dt)

        self.pose_constrain_L = attributes.get('constrain_L', False)

        # Below is an untested idea:
        self.delta_shrinkage = .8
        # initiate the Asymptotic control mode
        self.asymptotic_control_on = attributes.get('asymptotic_mode', True)
        if self.asymptotic_control_on:
            self.delta = attributes.get('delta_asymptotic_const', 0.2)

            # asymptotic control parameters sequence
            self.asym_control_phi = np.zeros(self.steps)
            self.asym_control_dphi = np.zeros(self.steps)

            # self.asym_control_ddphi = np.zeros(self.steps)
            self.asym_control_L = np.zeros(self.steps)
            self.asym_control_L[0] = self.l0
            self.asym_control_dL = np.zeros(self.steps)

        else:
            pass

        # initiate the Adaptive control mode
        self.adaptive_control_on = attributes.get('adaptive_mode', False)
        if self.adaptive_control_on:
            self.delta_adaptive_const = attributes.get('delta_adaptive_const', 0.15)
            self.delta = attributes.get('delta_adaptive_const', 0.2)

            # adaptive control parameters sequence
            self.adap_control_phi = np.zeros(self.steps)
            self.adap_control_dphi = np.zeros(self.steps)

            # self.adap_control_ddphi = np.zeros(self.steps)
            self.adap_control_length = np.zeros(self.steps)
            self.adap_control_dlength = np.zeros(self.steps)

            self.adap_delta_sequence = np.zeros(self.steps)
        else:
            pass

        self.no_control_on = attributes.get('no_control_mode', False)
        # oscillating control sequence
        if self.no_control_on:
            self.oscillating_phi = np.zeros(self.steps)
            self.oscillating_dphi = np.zeros(self.steps)
            self.oscillating_ddphi = np.zeros(self.steps)
        else:
            pass

        # The pendulum length and length derivative at the current time step
        self.L = self.l0
        self.dL = 0

        # The time marching scheme could be used, 1 = Forward Euler or 2 = RK4
        self.time_marching_method = attributes.get('time-marching scheme', 2)

        # The root folder path and images folder path
        self.ROOT_PATH = os.getcwd()
        self.IMG_PATH = os.path.join(self.ROOT_PATH, 'images')

        # plot setting
        self.plot_trigger = attributes.get('plot', False)
        self.ylim = np.max(self.wave_phi) * 1.1
        # save fig and show fig indicator
        self.save_fig = attributes.get('save_fig', False)
        self.show_fig = attributes.get('show_fig', False)
        # saving picture in designated format
        self.format_fig = attributes.get('format_fig', '.png')
        # save data
        self.save_data = attributes.get('save_data', True)

    # ====================================   EQUATION OF MOTION   ====================================
    def fixed_length_eom(self, x):
        '''
        ====================================
         ..
        phi = - g * sin(phi) / L;
        ====================================
        phi^n+1 - phi^n     .
        ---------------- = phi^n
               dt
         .         .
        phi^n+1 - phi^n
        ---------------- = -w^2 * phi^n
               dt
        ====================================
                                .
        phi^n+1 = phi^n + dt * phi^n
         .          .
        phi^n+1 = phi^n - dt * w^2 * phi^n
        ====================================
        :return:
        '''
        x_dot = np.empty((2, ))
        x_dot[0] = x[1]
        x_dot[1] = -self.frequency**2 * x[0]
        return x_dot

    def variable_length_eom(self, x):
        '''
        First compute the length for each of the states x;
        Second compute the time-change of the equation;
         ..         .    .
        phi = [-2 * L * phi - g * sin(phi)] / L
        states = [x0, x1]' = [phi, dphi]'
        :return:
        '''
        # Here we can't directly compute Ldot, otherwise it would fall into infinite loop.
        # Use Ldot = L0*delta*([d(phi)/dt]^2 + phi * d^2(phi)/dt^2), plug it back into the system dynamic
        L = self.compute_length(x)

        f = np.empty((2, ))
        f[0] = x[1]
        if abs(L + self.l0 * (1 + 3 * self.delta * x[0] * x[1])) > 1e-4:
            # Regarding the expression to compute f[1], use L instead of plugging L=L0[1+delta*phi*d(phi/dt)] directly,
            # easier to add constrain on L(t)
            f[1] = (-2 * self.l0 * self.delta * x[1]**3 - self.g * np.sin(x[0])) / \
                   (L + self.l0 * (1 + 3 * self.delta * x[0] * x[1]))
        else:
            f[1] = 0
            # f[1] = (-2 * self.l0 * self.delta * x[1]**3 - self.g * np.sin(x[0])) / \
            #        (L + self.l0 * (1 + 3 * self.delta_shrinkage * self.delta * x[0] * x[1]))
            print('Invalid values for denominator encountered in the computation of variable-length equation of motion')
            print('Set it to be 0 temporarily.')
        return f

    def compute_length(self, x):
        L = self.l0 * (1 + self.delta * x[0] * x[1])
        if self.pose_constrain_L:
            L = np.clip(L, self.Lmin, self.Lmax)
        else:
            pass
        return L

    def compute_length_dot(self, x):
        f = self.variable_length_eom(x)
        Ldot = self.l0 * self.delta * (x[1]**2 + x[0]*f[1])
        if self.pose_constrain_L:
            Ldot = np.clip(Ldot, self.Ldotmin, self.Ldotmax)
        else:
            pass
        return Ldot

    def delta_update(self, t):
        """
        Adaptive control; To accelerate the convergence of the pendulum, set the amplitude of control inputs consisted of
        pendulum length to be large enough
        :param t: time instance
        :return:
        """
        # combine controlled dphi with the dphi before control starts
        dphi_sequence = np.hstack((self.wave_dphi, self.adap_control_dphi[:t]))
        # 1st find the positions that dphi <= 1e-2/2
        dphi_zero_list = np.where(np.abs(dphi_sequence) <= 1e-2/2)[0]
        if len(dphi_zero_list) > 0:
            # 2nd find the last group of dphi <= 1e-2/2
            last_dphi_zero = utils.last_consecutives(dphi_zero_list)
            # 3rd find the largest amplitude of the angle in the last cycle
            phi_sequence = np.hstack((self.wave_phi, self.adap_control_phi[:t]))
            self.delta_adaptive = self.delta_adaptive_const / np.abs(np.max(phi_sequence[last_dphi_zero]))
        else:
            # dphi have not finished one peak yet
            self.delta_adaptive = self.delta_adaptive_const / 1
        self.adap_delta_sequence[t] = np.copy(self.delta_adaptive)

    # ====================================   MAIN SCRIPT   ====================================
    def free_pendulum_oscillation(self):
        '''

        Reference link:http://hplgit.github.io/Programming-for-Computations/pub/p4c/._p4c-solarized-Python022.html
        :return:
        '''
        for step, _ in enumerate(self.time):
            if step == 0:
                state = np.hstack((self.wave_phi[-1], self.wave_dphi[-1]))
            else:
                state = np.hstack((self.oscillating_phi[step - 1], self.oscillating_dphi[step - 1]))
            # time marching ODE
            self.oscillating_phi[step], self.oscillating_dphi[step] = self.time_marching(state, self.fixed_length_eom)

    def asymptotic_control_pendulum_oscillation(self):
        for i in range(self.steps):
            if i == 0:
                state = np.hstack((self.wave_phi[-1], self.wave_dphi[-1]))
            else:
                state = np.hstack((self.asym_control_phi[i-1], self.asym_control_dphi[i-1]))

            # update the length of the pendulum
            self.asym_control_L[i] = self.compute_length(state)
            self.asym_control_dL[i] = self.compute_length_dot(state)

            # time marching the system dynamic
            self.asym_control_phi[i], self.asym_control_dphi[i] = self.time_marching(state, self.variable_length_eom)

    def adaptive_control_pendulum_oscillation(self):
        for step, _ in enumerate(self.time):
            if step == 0:  # assemble state -> ODE, full_state -> length
                state = np.hstack((self.wave_phi[-1], self.wave_dphi[-1]))
                full_state = np.hstack((self.wave_phi[-1], self.wave_dphi[-1], self.wave_ddphi[-1]))
            else:
                state = np.hstack((self.adap_control_phi[step - 1], self.adap_control_dphi[step - 1]))
                full_state = np.hstack(
                    (self.adap_control_phi[step - 1], self.adap_control_dphi[step - 1], self.adap_control_ddphi[step - 1]))
            # update delta
            self.delta_update(step)

            # update length <- phi, phi_dot, phi_double_dot
            self.adap_control_length[step], self.adap_control_dlength[step] = self.length_update(full_state)

            # assemble length vector
            length = np.hstack((self.adap_control_length[step], self.adap_control_dlength[step]))

            # update phi_double_dot, this is for updating the length at the next time step
            self.adap_control_ddphi[step] = self.update_ddphi(state, length)

            # assemble the ODE
            func = partial(self.variable_length_eom, input=length)

            # time marching ODE
            self.adap_control_phi[step], self.adap_control_dphi[step] = self.time_marching(state, func)

    def main(self):
        if self.no_control_on:
            self.free_pendulum_oscillation()
        else:
            pass

        # invoke the asymptotic control
        if self.asymptotic_control_on:
            self.asymptotic_control_pendulum_oscillation()
        else:
            pass

        # invoke the adaptive control
        if self.adaptive_control_on:
            self.adaptive_control_pendulum_oscillation()
        else:
            pass

        # plot trigger
        if self.plot_trigger:
            self.plot()
        else:
            pass

        # save data
        if self.save_data:
            self.data_saver()

    # ====================================   TIME MARCHING SCHEME   ====================================

    def forward_euler(self, x, func):
        x_dot = func(x)
        x_new = x + x_dot * self.dt
        return x_new

    def runge_kutta4(self, x, func):
        f1 = func(x)
        f2 = func(x + self.dt/2 * f1)
        f3 = func(x + self.dt/2 * f2)
        f4 = func(x + self.dt * f3)
        x_new = x + self.dt/6 * (f1 + 2*f2 + 2*f3 + f4)
        return x_new

    def time_marching(self, x, func):
        if self.time_marching_method == 1:
            # Forward Euler
            x_new = self.forward_euler(x, func)
        elif self.time_marching_method == 2:
            # RK4
            x_new = self.runge_kutta4(x, func)
        else:
            # new scheme needs to be claimed
            x_new = np.zeros(2)
            raise ValueError('New time-marching scheme need to specified. Otherwise set "time-marching scheme"=2.')
        return x_new[0], x_new[1]

    # ====================================   PLOT   ====================================
    def plot(self):
        self.phi_plot()
        self.phase_space_plot()
        self.length_plot()
        # self.delta_plot()
        # self.phi_dphi_time_plot()
        self.energy_plot()
        # self.frequency_plot()

    def phi_plot(self):
        plt.figure(figsize=[16, 9])
        plt.grid()
        self.control_indicator_plot()
        if self.no_control_on:
            self.free_pendulum_plot()
        self.control_pendulum_plot()
        plt.legend()
        self.fig_save_and_show('phi')

    def control_indicator_plot(self):
        y = np.linspace(-1 * self.ylim, self.ylim, 200)
        t = self.control_start_time * np.ones(y.shape[0])
        plt.plot(t, y, 'g-.', label=r'$Control\ starts$')

    def free_pendulum_plot(self):
        entire_free_phi = np.hstack((self.wave_phi, self.oscillating_phi))
        plt.plot(self.entire_t, entire_free_phi, 'b-', label=r'$\phi(t) - No Control$')
        plt.xlabel(r'$t$', fontsize=21)
        plt.ylabel(r'$\phi(t)$', fontsize=21, rotation=0)

    def control_pendulum_plot(self):
        control_t = np.arange(0, self.max_t, self.dt) + self.control_start_time
        if self.asymptotic_control_on:
            plt.plot(control_t, self.asym_control_phi, 'r--', label=r'$\phi(t)-Controlled$')

        if self.adaptive_control_on:
            plt.plot(control_t, self.adap_control_phi, 'k-.', label=r'$\phi(t)-Adaptive\ \delta$')

    def phase_space_plot(self):
        # 2D phase plot
        plt.figure(figsize=[16, 9])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.scatter(self.asym_control_phi[0], self.asym_control_dphi[0], c='b', marker='s', label='initial')
        asym_x_range = asym_y_range = adap_x_range = adap_y_range = np.ones(1)

        if self.asymptotic_control_on:
            # plot the asymptotic convergence of phi(t)
            plt.plot(self.asym_control_phi, self.asym_control_dphi, 'r--', label='Asymptotic')
            self.phase_space_arrow_plot(self.asym_control_phi, self.asym_control_dphi)
            asym_x_range = np.ptp(self.asym_control_phi, axis=0)
            asym_y_range = np.ptp(self.asym_control_dphi, axis=0)

        if self.adaptive_control_on:
            # plot the exponential convergence of phi(t)
            plt.plot(self.adap_control_phi, self.adap_control_dphi, 'k--', label='Adaptive')
            self.phase_space_arrow_plot(self.asym_control_phi, self.asym_control_dphi)
            adap_x_range = np.ptp(self.adap_control_phi, axis=0)
            adap_y_range = np.ptp(self.adap_control_dphi, axis=0)

        r = np.max(np.hstack((adap_y_range, asym_y_range, adap_x_range, asym_x_range))) * 1.2

        plt.ylim(0 - r / 2, 0 + r/2)
        plt.xlim(0 - r / 2, 0 + r/2)

        plt.xlabel(r'$\phi(t)$', fontsize=21)
        plt.ylabel(r'$\dot{\phi}(t)$', fontsize=21, rotation=0)
        plt.legend()
        self.fig_save_and_show('phi_dphi')

    def phase_space_arrow_plot(self, phi, dphi):
        # plot 10 arrows for every 100/1000 steps; if the steps < 100, plot just one arrow
        if self.steps < 100:
            num_arrows = 1
            arrow_position_sequence = np.array([int(round(self.steps/2))])
        else:
            num_arrows = 5
            arrow_position_sequence = np.array([int(step) for step in np.linspace(0, self.steps - 2, num_arrows)])

        for num, pos in enumerate(arrow_position_sequence[1:]):
            plt.annotate('', xy=(phi[pos+1], dphi[pos+1]), xytext=(phi[pos], dphi[pos]),
                         arrowprops={'arrowstyle': '->, head_width=.5, head_length=1.5', 'color': 'r'}, va='center')

    def phi_dphi_time_plot(self):
        # Debug use
        if self.asymptotic_control_on:
            plt.figure(figsize=[16, 9])
            plt.grid()
            plt.plot(self.time, self.asym_control_phi, 'r', label=r'$\phi(t)$')
            plt.plot(self.time, self.asym_control_dphi, 'b', label=r'$\dot{\phi}(t)$')
            plt.xlabel(r'$t$', size=21)
            plt.legend()
            self.fig_save_and_show('phi_dphi_time_asym')

        if self.adaptive_control_on:
            plt.figure(figsize=[16, 9])
            plt.grid()
            plt.plot(self.time, self.adap_control_phi, 'r', label=r'$\phi(t)$')
            plt.plot(self.time, self.adap_control_dphi, 'b', label=r'$\dot{\phi}(t)$')
            plt.xlabel(r'$t$', size=21)
            plt.legend()
            self.fig_save_and_show('phi_dphi_time_adap')

    def energy_plot(self):
        plt.figure(figsize=[16, 9])
        plt.grid()
        if self.asymptotic_control_on:
            energy_asym = 1/2 * self.m * ((self.asym_control_L * self.asym_control_dphi)**2 + self.asym_control_dL**2) \
                          + self.m * self.g * (-np.cos(self.asym_control_phi) * self.asym_control_L + self.l0)
            plt.plot(self.time, energy_asym, 'r', label=r'$V$ - Asymptotic')

        if self.adaptive_control_on:
            energy_adap = 1/2 * ((self.adap_control_length * self.adap_control_dphi)**2 + self.adap_control_dlength**2) \
                          + self.g*self.adap_control_length*(1-np.cos(self.adap_control_phi))
            plt.plot(self.time, energy_adap, 'k--', label=r'$V$ - Adaptive')
        plt.xlabel(r'$t$', size=21)
        plt.ylabel(r'$V(t)$', size=21, rotation=0, labelpad=22)
        plt.legend()
        self.fig_save_and_show('energy')

    def frequency_plot(self):
        plt.figure(figsize=[16, 9])
        plt.grid()
        if self.asymptotic_control_on:
            omega_asym = np.sqrt(self.g / self.asym_control_L - (self.asym_control_dL/self.asym_control_L)**2)
            plt.plot(self.time, omega_asym, 'r', label=r'$\omega$ - Asymptotic')

        if self.adaptive_control_on:
            omega_adap = np.sqrt(self.g / self.adap_control_length - (self.adap_control_dlength/self.adap_control_length)**2)
            plt.plot(self.time, omega_adap, 'k--', label=r'$\omega$- Adaptive')
        plt.xlabel(r'$t$', size=21)
        plt.ylabel(r'$\omega(t)$', size=21, rotation=0)
        plt.legend()
        self.fig_save_and_show('frequency')

    def length_plot(self):
        # length plot
        plt.figure(figsize=[16, 9])
        plt.grid()

        # L plot
        if self.asymptotic_control_on:
            # plot the asymptotic pendulum length
            entire_asym_length = np.hstack((self.l0 * np.ones(self.prev_length), self.asym_control_L))
            plt.plot(self.entire_t, entire_asym_length, 'r--', label='Asymptotic')

        if self.adaptive_control_on:
            # plot the adaptive pendulum length
            entire_adap_length = np.hstack((self.l0 * np.ones(self.prev_length), self.adap_control_length))
            plt.plot(self.entire_t, entire_adap_length, 'k-.', label='Adaptive')
        plt.xlabel(r'$t$', fontsize=21)
        plt.ylabel(r'$\frac{L(t)}{L_0}$', fontsize=21, rotation=0, labelpad=18)
        plt.legend()
        self.fig_save_and_show('length')

        # dL plot
        plt.figure(figsize=[16, 9])
        plt.grid()

        if self.asymptotic_control_on:
            # plot the asymptotic pendulum dlength
            entire_asym_dlength = np.hstack((np.zeros(self.prev_length), self.asym_control_dL))
            plt.plot(self.entire_t, entire_asym_dlength, 'r--', label='Asymptotic')
        if self.adaptive_control_on:
            # plot the adaptive pendulum length
            entire_adap_dlength = np.hstack((np.zeros(self.prev_length), self.adap_control_dlength))
            plt.plot(self.entire_t, entire_adap_dlength, 'k-.', label='Adaptive')
        plt.xlabel(r'$t$', fontsize=21)
        plt.ylabel(r'$\dot{L}(t)$', fontsize=21, rotation=0, labelpad=18)
        plt.legend()
        self.fig_save_and_show('dlength')

    def delta_plot(self):
        plt.figure(figsize=[16, 9])
        plt.grid()

        if self.asymptotic_control_on:
            # plot the asymptotic delta (const)
            entire_asym_delta = self.delta_asymptotic * np.ones(self.entire_t.shape[0])
            plt.plot(self.entire_t, entire_asym_delta, 'r--', label='Asymptotic')

        if self.adaptive_control_on:
            entire_adap_delta = np.hstack((np.zeros(self.prev_length), self.adap_delta_sequence))
            plt.plot(self.entire_t, entire_adap_delta, 'k-.', label='Adaptive')
        plt.legend()
        plt.xlabel(r'$t$', fontsize=21)
        plt.ylabel(r'$\delta(t)$', fontsize=21, rotation=0)
        self.fig_save_and_show('delta')

    def fig_save_and_show(self, name):
        if self.save_fig:
            name = os.path.join(self.IMG_PATH, name) + self.format_fig
            plt.savefig(name, format=self.format_fig[1:], dpi=300)
        if self.show_fig:
            plt.show()

    def data_saver(self):
        data = {
            'asym_phi': self.asym_control_phi,
            'asym_dphi': self.asym_control_dphi,
            'asym_L': self.asym_control_L,
            'asym_dL': self.asym_control_dL
        }
        io.savemat('pendulum_data.mat', data)


if __name__ == "__main__":
    # T: total control time
    T = 10
    dt = 0.02
    g = 9.8
    m = 1
    l = 1

    # t_length: time length before control starts
    t_length = dt
    t = np.arange(0, t_length, dt)

    # 1st: simple sin wave
    # signal = {
    #     'phi': a * np.sin(w0 * t),
    #     'dphi': a * w0 * np.cos(w0 * t),
    # }

    signal = {
        'phi': -2 * np.ones(1),
        'dphi': 3 * np.ones(1),
    }

    properties = {
        'm': m,
        'max_t': T,
        'dt': dt,
        'adaptive_mode': False,
        'delta_adaptive_const': .15,
        'asymptotic_mode': True,
        'delta_asymptotic_const': .05,
        'l0': l,
        'constrain_L': True,
        'Ldotmax': 5,
        'Ldotmin': -5,
        'Lmax': 1.5,
        'Lmin': 0.5,
        'g': g,
        'plot': True,
        'save_fig': True,
        'show_fig': False,
        'save_data': False,
    }
    # ================
    pendu = Pendulum(signal, properties)
    pendu.main()
