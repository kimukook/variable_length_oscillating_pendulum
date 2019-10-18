import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


'''
This is a script that simulates the movement of pendulum.
equation of motion:

   ..           .    .
  theta = [-2 * L * phi - g * sin(theta)] / l

'''


class Pendulum:
    def __init__(self, wave, attributes):
        self.max_t = attributes.get('max_t', 30)
        self.dt = attributes.get('dt', 0.01)
        self.steps = int(self.max_t / self.dt)
        self.time = np.linspace(0, self.max_t, self.steps)

        self.l0 = attributes.get('l0', 1)
        self.Ldotmax = attributes.get('Ldotmax', 1)
        self.Ldotmin = attributes.get('Ldotmin', -1)

        self.Lmax = attributes.get('Lmax', 1.2 * self.l0)
        self.Lmin = attributes.get('Lmin', .8 * self.l0)

        self.g = 9.8

        self.wave_phi = wave['phi']
        self.wave_dphi = wave['dphi']
        self.wave_ddphi = wave['ddphi']
        self.amplitude = wave.get('amplitude', 1)
        self.frequency = wave.get('frequency', np.pi/2)
        self.control_start_time = self.dt * self.wave_phi.shape[0]
        self.prev_length = self.wave_phi.shape[0]
        self.entire_t = np.arange(0, self.dt * self.prev_length + self.max_t, self.dt)

        self.delta = 0
        if attributes['delta_mode_adaptive']:
            self.delta_adaptively_control_on = True
            self.delta_adaptive_const = attributes.get('delta_adaptive_const', 0.15)
            self.delta_adaptive = 0
        else:
            self.delta_adaptively_control_on = False
        self.delta_adaptive_working = False

        if attributes['delta_mode_asymptotic']:
            self.delta_asymptotically_control_on = True
            self.delta_asymptotic = attributes.get('delta_asymptotic_const', 0.1)
        else:
            self.delta_asymptotically_control_on = False
        self.delta_asymptotic_working = False

        self.oscillating_phi = np.zeros(self.steps)
        self.oscillating_dphi = np.zeros(self.steps)
        self.oscillating_ddphi = np.zeros(self.steps)

        self.asym_control_phi = np.zeros(self.steps)
        self.asym_control_dphi = np.zeros(self.steps)
        self.asym_control_ddphi = np.zeros(self.steps)
        self.asym_control_length = np.zeros(self.steps)
        self.asym_control_dlength = np.zeros(self.steps)

        self.adap_control_phi = np.zeros(self.steps)
        self.adap_control_dphi = np.zeros(self.steps)
        self.adap_control_ddphi = np.zeros(self.steps)
        self.adap_control_length = np.zeros(self.steps)
        self.adap_control_dlength = np.zeros(self.steps)
        self.adap_delta_sequence = np.zeros(self.steps)

        self.L = self.l0
        self.dL = 0
        self.time_marching_method = 2

        self.ROOT_PATH = os.getcwd()
        self.IMG_PATH = os.path.join(self.ROOT_PATH, 'images')
        self.ylim = self.amplitude * 1.1
        self.save_fig = attributes.get('save_fig', True)
        self.show_fig = attributes.get('show_fig', True)
        self.format_fig = '.png'

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

    def variable_length_eom(self, x, input):
        '''
         ..         .    .
        phi = [-2 * L * phi - g * sin(phi)] / L
        states = [x0, x1]' = [phi, dphi]'
        input = [L, dL]'
        :return:
        '''
        x_dot = np.empty((2, ))
        x_dot[0] = x[1]
        x_dot[1] = (-2 * input[1] * x[1] - self.g * np.sin(x[0])) / input[0]
        return x_dot

    def length_update(self, x):
        if self.delta_asymptotic_working:
            self.delta = self.delta_asymptotic
        elif self.delta_adaptive_working:
            self.delta = self.delta_adaptive
        else:
            raise ValueError('Neither asymptotic nor adaptive works now!')

        L = self.l0 * (1 + self.delta * x[0] * x[1])
        Ldot = self.l0 * self.delta * (x[1]**2 + x[0]*x[2])
        # L and Ldot constraints bound
        L = np.clip(L, self.Lmin, self.Lmax)
        if L == self.Lmin or L == self.Lmax:
            Ldot = 0
        else:
            Ldot = np.clip(Ldot, self.Ldotmin, self.Ldotmax)
        return L, Ldot

    # def length_update(self, x, L):
    #     if self.delta_asymptotic_working:
    #             self.delta = self.delta_asymptotic
    #         elif self.delta_adaptive_working:
    #             self.delta = self.delta_adaptive
    #         else:
    #             raise ValueError('Neither asymptotic nor adaptive works now!')
    #
    #     Ldot = self.l0 * self.delta * (x[1]**2 + x[0]*x[2])
    #     # L and Ldot constraints bound
    #     Ldot = np.clip(Ldot, self.Ldotmin, self.Ldotmax)
    #     L += self.dt * Ldot
    #     L = np.clip(L, self.Lmin, self.Lmax)
    #     if L == self.Lmin or L == self.Lmax:
    #         Ldot = 0
    #     return L, Ldot

    def delta_update(self, t):
        # combine dphi with dphi before control starts
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

    def update_ddphi(self, state, length):
        return (-2 * length[1] * state[1] - self.g * np.sin(state[0])) / length[0]

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

    def control_pendulum_oscillation(self):
        self.delta_asymptotic_working = True
        for step, _ in enumerate(self.time):
            if step == 0:  # assemble state -> ODE, full_state -> length
                state = np.hstack((self.wave_phi[-1], self.wave_dphi[-1]))
                full_state = np.hstack((self.wave_phi[-1], self.wave_dphi[-1], self.wave_ddphi[-1]))
            else:
                state = np.hstack((self.asym_control_phi[step - 1], self.asym_control_dphi[step - 1]))
                full_state = np.hstack((self.asym_control_phi[step - 1], self.asym_control_dphi[step - 1],
                                        self.asym_control_ddphi[step - 1]))

            # update length <- phi, phi_dot, phi_double_dot
            self.asym_control_length[step], self.asym_control_dlength[step] = self.length_update(full_state)

            # update length <- phi, phi_dot, phi_double_dot
            length = np.hstack((self.asym_control_length[step], self.asym_control_dlength[step]))

            # update phi_double_dot, this is for updating the length at the next time step
            self.asym_control_ddphi[step] = self.update_ddphi(state, length)

            # assemble the ODE
            func = partial(self.variable_length_eom, input=length)

            # time marching ODE
            self.asym_control_phi[step], self.asym_control_dphi[step] = self.time_marching(state, func)

        self.delta_asymptotic_working = False

    def adaptive_control_pendulum_oscillation(self):
        self.delta_adaptive_working = True
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

        self.delta_adaptive_working = False

    # def main(self):

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
        return x_new[0], x_new[1]

    # ====================================   PLOT   ====================================
    def plot(self):
        self.phi_plot()
        self.phi_dphi_plot()
        self.length_plot()
        self.delta_plot()

    def phi_plot(self):
        plt.figure(figsize=[16, 9])
        plt.grid()
        self.control_indicator_plot()
        self.free_pendulum_plot()
        self.control_pendulum_plot()
        plt.legend()
        self.fig_save_and_show('phi')

    def control_indicator_plot(self):
        y = np.linspace(-self.ylim, self.ylim, 200)
        t = self.control_start_time * np.ones(y.shape[0])
        plt.plot(t, y, 'g-.', label=r'$Control\ starts$')

    def free_pendulum_plot(self):
        entire_free_phi = np.hstack((self.wave_phi, self.oscillating_phi))
        plt.plot(self.entire_t, entire_free_phi, 'b-', label=r'$\phi(t) - No Control$')
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$\phi(t)$', fontsize=18)

    def control_pendulum_plot(self):
        control_t = np.arange(0, self.max_t, self.dt) + self.control_start_time
        if self.delta_asymptotically_control_on:
            plt.plot(control_t, self.asym_control_phi, 'r--', label=r'$\phi(t)-Controlled$')
        if self.delta_adaptively_control_on:
            plt.plot(control_t, self.adap_control_phi, 'k-.', label=r'$\phi(t)-Adaptive\ \delta$')

    def phi_dphi_plot(self):
        plt.figure(figsize=[16, 9])
        plt.grid()
        if self.delta_asymptotically_control_on:
            # plot the asymptotic convergence of phi(t)
            plt.plot(self.asym_control_phi, self.asym_control_dphi, 'r--', label='Asymptotic')

        if self.delta_adaptively_control_on:
            # plot the exponential convergence of phi(t)
            plt.plot(self.adap_control_phi, self.adap_control_dphi, 'k-.', label='Adaptive')
        plt.xlabel(r'$\phi(t)$', fontsize=18)
        plt.ylabel(r'$\dot{\phi}(t)$', fontsize=18)
        plt.legend()
        self.fig_save_and_show('phi_dphi')

    def length_plot(self):
        # length plot
        plt.figure(figsize=[16, 9])
        plt.grid()

        # L plot
        if self.delta_asymptotically_control_on:
            # plot the asymptotic pendulum length
            entire_asym_length = np.hstack((self.l0 * np.ones(self.prev_length), self.asym_control_length))
            plt.plot(self.entire_t, entire_asym_length, 'r--', label='Asymptotic')
        if self.delta_adaptively_control_on:
            # plot the adaptive pendulum length
            entire_adap_length = np.hstack((self.l0 * np.ones(self.prev_length), self.adap_control_length))
            plt.plot(self.entire_t, entire_adap_length, 'k-.', label='Adaptive')
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$L(t)/L_0$', fontsize=18)
        plt.legend()
        self.fig_save_and_show('length')

        # dL plot
        plt.figure(figsize=[16, 9])
        plt.grid()

        if self.delta_asymptotically_control_on:
            # plot the asymptotic pendulum dlength
            entire_asym_dlength = np.hstack((np.zeros(self.prev_length), self.asym_control_dlength))
            plt.plot(self.entire_t, entire_asym_dlength, 'r--', label='Asymptotic')
        if self.delta_adaptively_control_on:
            # plot the adaptive pendulum length
            entire_adap_dlength = np.hstack((np.zeros(self.prev_length), self.adap_control_dlength))
            plt.plot(self.entire_t, entire_adap_dlength, 'k-.', label='Adaptive')
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$\dot{L}(t)$', fontsize=18)
        plt.legend()
        self.fig_save_and_show('dlength')

    def delta_plot(self):
        plt.figure(figsize=[16, 9])
        plt.grid()

        if self.delta_asymptotically_control_on:
            # plot the asymptotic delta (const)
            entire_asym_delta = self.delta_asymptotic * np.ones(self.entire_t.shape[0])
            plt.plot(self.entire_t, entire_asym_delta, 'r--', label='Asymptotic')

        if self.delta_adaptively_control_on:
            entire_adap_delta = np.hstack((np.zeros(self.prev_length), self.adap_delta_sequence))
            plt.plot(self.entire_t, entire_adap_delta, 'k-.', label='Adaptive')
        plt.legend()
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$\delta(t)$', fontsize=18)
        self.fig_save_and_show('delta')

    def fig_save_and_show(self, name):
        if self.save_fig:
            name = os.path.join(self.IMG_PATH, name) + self.format_fig
            plt.savefig(name, format=self.format_fig[1:], dpi=300)
        if self.show_fig:
            plt.show()


if __name__ == "__main__":
    a = 2
    T = 20
    dt = 0.001
    g = 9.8
    l = 1
    w0 = np.sqrt(g / l)
    t_length = dt * 130 * 20
    t = np.arange(0, t_length, dt)

    wave = {
        'amplitude': a,
        'frequency': w0,
        'phi': a * np.sin(w0 * t) + a * np.cos(w0 * t),
        'dphi': a * w0 * np.cos(w0 * t) - a * w0 * np.sin(w0 * t),
        'ddphi': -a * w0**2 * np.sin(w0 * t) - a * w0**2 * np.cos(w0 * t)
    }

    attributes = {
        'max_t': T,
        'dt': dt,
        'save_fig': True,
        'show_fig': False,
        'delta_mode_adaptive': True,
        'delta_adaptive_const': .15,
        'delta_mode_asymptotic': True,
        'delta_asymptotic_const': .1
    }
    # ================
    pendu = Pendulum(wave, attributes)
    pendu.free_pendulum_oscillation()
    pendu.control_pendulum_oscillation()
    pendu.adaptive_control_pendulum_oscillation()
    pendu.plot()

    # DEBUG
    # t_mu = np.arange(0, 10, dt)
    # test_phi = a * np.sin(w0 * t_mu)
    # test_dphi = a * w0 * np.cos(w0 * t_mu)
    # test_ddphi = -a * w0**2 * np.sin(w0 * t_mu)
    # count = np.arange(0, 130*dt, dt).shape[0] + step

    # compare:
    # test_phi[count], test_dphi[count]
    # self.adap_control_phi[step], self.adap_control_dphi[step]




