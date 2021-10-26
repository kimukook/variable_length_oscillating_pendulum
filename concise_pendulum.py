import numpy as np
import matplotlib.pyplot as plt
'''
This piece of code was originated from Tom's code in MATLAB
'''


class PendulumParams:
    def __init__(self, m, g, L0, delta, Lmax, Lmin):
        self.m = m
        self.g = g
        self.L0 = L0
        self.delta = delta
        self.Lmax = Lmax
        self.Lmin = Lmin


class Pendulum:
    def __init__(self, pendulum_params):
        self.p = pendulum_params

    def RHS(self, x):
        f = np.zeros(2)
        f[0] = x[1]
        f[1] = (-self.p.g*np.sin(x[0]) - 2*self.p.L0*self.p.delta*x[1]**3)/(self.length(x)+2*self.p.L0*self.p.delta*x[0]*x[1])
        return f

    def length(self, x):
        L = self.p.L0 * (1 + self.p.delta * x[0] * x[1])
        # L = np.clip(L, self.p.Lmin, self.p.Lmax)
        return L

    def length_dot(self, x):
        f = self.RHS(x)
        Ldot = self.p.L0 * self.p.delta * (x[1]**2 + x[0] * f[1])
        return Ldot


if __name__ == '__main__':
    m = 1
    g = 9.8
    L0 = 1
    delta = 0.05
    Lmax = 1.5
    Lmin = 0.5
    params = PendulumParams(m, g, L0, delta, Lmax, Lmin)
    pendulum = Pendulum(params)

    Tmax = 10
    h = .02
    steps = int(Tmax/h)
    x = np.zeros((2, steps))
    x[0, 0] = 2
    x[1, 0] = 1

    p_length = np.zeros(steps)
    p_length[0] = params.L0

    for i in range(steps-1):
        L = pendulum.length(x[:, i])
        p_length[i+1] = np.copy(L)
        f1 = pendulum.RHS(x[:, i])
        f2 = pendulum.RHS(x[:, i] + h * f1 / 2)
        f3 = pendulum.RHS(x[:, i] + h * f2 / 2)
        f4 = pendulum.RHS(x[:, i] + h * f3)
        x[:, i+1] = x[:, i] + h*(f1+2*f2+2*f3+f4)/6

    t = np.linspace(0, Tmax, steps)

    fig = plt.figure(figsize=[16, 9])
    plt.plot(t, x[0, :], 'b--')
    plt.grid()
    plt.xlabel('t', size=20)
    plt.ylabel(r'$\phi(t)$', size=20, rotation=0)
    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=[16, 9])
    plt.plot(t, p_length, 'k--')
    plt.grid()
    plt.xlabel('t', size=20)
    plt.ylabel(r'$L(t)$', size=20, rotation=0)
    plt.show()
    plt.close(fig)
