import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import os
import concise_pendulum
from os.path import dirname, join


class NeuralNetworkLyapunovFunctionPlot:
    def __init__(self, range, step_size):
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        self.w3 = None
        self.b3 = None

        self.step_size = step_size
        self.range = range

        self.V = None
        self.Vdot = None

        self.V_eval = None

        m = 1
        g = 9.8
        L0 = 1
        delta = 0.05
        Lmax = 1.5
        Lmin = 0.5
        params = concise_pendulum.PendulumParams(m, g, L0, delta, Lmax, Lmin)
        self.system = concise_pendulum.Pendulum(params)

    def set_nn(self, w1, w2, b1, b2, w3=None, b3=None):
        if w1 is not None:
            self.w1 = w1
        if w2 is not None:
            self.w2 = w2.reshape(-1, 1)
        if w3 is not None:
            self.w3 = w3.reshape(-1, 1)

        if b1 is not None:
            self.b1 = b1.reshape(-1, 1)
        if b2 is not None:
            self.b2 = b2
        if b3 is not None:
            self.b3 = b3

    def nnlf_tanh_eval(self, x):
        hidden = self.w1 @ x + self.b1
        output = self.w2.T @ np.tanh(hidden) + self.b2
        return np.tanh(output)

    def nnlf_tanh_eval3(self, x):
        hidden1 = self.w1 @ x + self.b1
        hidden2 = self.w2 @ np.tanh(hidden1) + self.b2
        output = self.w3 @ np.tanh(hidden2) + self.b3
        return np.tanh(output)

    def dtanh(self, x):
        return 1 - x**2

    def nnlf_square_eval(self, x):
        hidden = self.w1 @ x + self.b1
        output = self.w2.T @ np.tanh(hidden) + self.b2
        return output**2

    def dsquare(self, x):
        return np.sqrt(x)*2

    def f_eval(self, x):
        f = np.zeros((2, 1))
        f[0] = x[1]
        f[1] = -(self.system.p.g * np.sin(x[0]) + 2 * self.system.p.L0 * self.system.p.delta * x[1]**3) / \
               (self.system.length(x) + 2 * self.system.p.L0 * self.system.p.delta * x[0] * x[1])
        return f

    def plot(self):
        x, y = np.linspace(-self.range, self.range, self.step_size), np.linspace(-self.range, self.range, self.step_size)
        X, Y = np.meshgrid(x, y)

        self.V = np.zeros(X.shape)
        self.Vdot = np.zeros(X.shape)

        for i in range(self.step_size):
            for j in range(self.step_size):
                point = np.array([X[i, j], Y[i, j]]).reshape(-1, 1)
                self.V[i, j] = self.nnlf_tanh_eval(point)
                self.Vdot[i, j] = self.dtanh(self.V[i, j]) * self.w2.T @ (self.dtanh(np.tanh(self.w1 @ point + self.b1)) * self.w1) @ self.f_eval(point)
                # self.V[i, j] = self.nnlf_square_eval(point)
                # self.Vdot[i, j] = self.dsquare(self.V[i, j]) * self.w2.T @ (self.dtanh(np.tanh(self.w1 @ point + self.b1)) * self.w1) @ self.f_eval(point)

        # plot V
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.contour(X, Y, self.Vdot, zdir='z', offset=0, cmap=cm.coolwarm)
        ax.plot_surface(X, Y, self.V, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
        plt.xlabel(r'$\theta$', fontweight='bold', position=(0.5, 1))
        plt.ylabel(r'$\dot{\theta}$', fontweight='bold', position=(0.5, 0))
        ax.axes.set_zlim3d(bottom=0, top=.5)
        ax.set_zlabel('$V$')
        plt.title('Lyapunov Function')
        # plt.show()
        plt.savefig('NNLF_V.png', format='png', dpi=300)

        # from x axis view:
        ax.view_init(elev=0, azim=0)
        ax.set_xticks([])
        # plt.show()
        plt.savefig('NNLF_V2.png', format='png', dpi=300)
        plt.close(fig)

        # plot Vdot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.contour(X, Y, self.Vdot, zdir='z', offset=-2, cmap=cm.coolwarm)
        ax.plot_surface(X, Y, self.Vdot, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
        plt.xlabel(r'$\theta$', fontweight='bold', position=(0.5, 1))
        plt.ylabel(r'$\dot{\theta}$', fontweight='bold', position=(0.5, 0))
        ax.set_zlabel('$\dot{V}$')
        ax.axes.set_zlim3d(bottom=-2, top=.5)

        plt.title('The orbital derivative of Lyapunov Function')
        # plt.show()
        plt.savefig('NNLF_Vdot.png', format='png', dpi=300)

        ax.view_init(elev=0, azim=0)
        ax.set_xticks([])
        plt.savefig('NNLF_Vdot2.png', format='png', dpi=300)
        plt.close(fig)

        # Vdot from x axis
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.contour(X, Y, self.Vdot, zdir='z', offset=-2, cmap=cm.coolwarm)
        ax.plot_surface(X, Y, self.Vdot, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
        plt.xlabel(r'$\theta$', fontweight='bold', position=(0.5, 1))
        plt.ylabel(r'$\dot{\theta}$', fontweight='bold', position=(0.5, 0))
        ax.set_zlabel('$\dot{V}$')
        ax.axes.set_zlim3d(bottom=-2, top=.5)

        ax.view_init(elev=0, azim=90)
        ax.set_yticks([])
        plt.savefig('NNLF_Vdot3.png', format='png', dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    ROOT = dirname(dirname(os.getcwd()))
    # run code
    w1 = np.loadtxt(join(ROOT, "w1.txt"))
    w2 = np.loadtxt(join(ROOT, "w2.txt"))
    b1 = np.loadtxt(join(ROOT, "b1.txt"))
    b2 = np.loadtxt(join(ROOT, "b2.txt"))

    # python console
    # w1 = np.loadtxt("w1.txt")
    # w2 = np.loadtxt("w2.txt")
    # b1 = np.loadtxt("b1.txt")
    # b2 = np.loadtxt("b2.txt")

    plot_class = NeuralNetworkLyapunovFunctionPlot(1, 100)
    plot_class.set_nn(w1, w2, b1, b2)
    plot_class.plot()

