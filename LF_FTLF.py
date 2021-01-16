import numpy as np
import sympy as sp
from sympy import Symbol, Matrix, integrate, sin
import numdifftools as nd
from scipy.linalg import solve_continuous_lyapunov, sqrtm, inv, expm
import matplotlib.pyplot as plt
'''
This is a script that computes the Finite-time Lyapunov function for the variable-length pendulum.

=====================================
Author  :  Muhan Zhao
Date    :  Dec. 28, 2019
Location:  UC San Diego, La Jolla, CA
=====================================
0.  System model is given by
 
1.  Compute the linearization of the model at the equilibrium point

2.  Calculate the time shift d: Eqn (30) from Lazar's TAC2017, ||e^{d A}|| < 1, here A is the first order derivative 
    of f, at phi = 0.

    Transform this inequality into a minimization problem,

    min d
    s.t. d > 0 and ||e^{d A}|| -1 < 0




'''


class PendulumParams:
    def __init__(self, m, g, L0, delta, Lmax, Lmin):
        self.m = m
        self.g = g
        self.L0 = L0
        self.delta = delta
        self.Lmax = Lmax
        self.Lmin = Lmin


def length(x, p):
    L = p.L0 * (1 + p.delta * x[0] * x[1])
    L = np.clip(L, p.Lmin, p.Lmax)
    return L


def RHS(x, p):
    f = np.zeros(2)
    f[0] = x[1]
    f[1] = (-p.g*np.sin(x[0]) - 2*p.L0*p.delta*x[1]**3)/(length(x, p)+2*p.L0*p.delta*x[0]*x[1])
    return f


def weighted_mu(A, P):
    wp = sqrtm(P)@A@inv(sqrtm(P))
    mx = (1/2) * ( wp + wp.T )
    return np.max(np.linalg.eig(mx)[0])



def compute_d(A, P, dcs = np.linspace(5,0.1,100)):
    # TODO what is the norm using here?
    flag = False
    d = 0
    for dc in dcs:
        wp = sqrtm(P)@expm(dc*A)@inv(sqrtm(P))
        mx = (1/2) * (wp + wp.T)
        if np.max(np.linalg.eig(mx)[0]) < 1:
            d = dc
            flag = True
    if not flag:
        print('No d found, terminating.')
    return d


m = 1
g = 9.8
L0 = 1
delta = 0.05
Lmax = 1.5
Lmin = 0.5
p = PendulumParams(m, g, L0, delta, Lmax, Lmin)

f_ode = lambda x: RHS(x, p)
Dfx = nd.Jacobian(f_ode)
E = np.array([0, 0])
A = Dfx(E)

I = np.identity(2)
P = solve_continuous_lyapunov(A, -10*I)


muA = weighted_mu(A, P)
d = compute_d(A, P)

pointsx = np.empty(shape=[1, 0])
pointsy = np.empty(shape=[1, 0])
pointsW = np.empty(shape=[1, 0])
pointsWdot = np.empty(shape=[1, 0])

x1 = Symbol('x1')
x2 = Symbol('x2')
t = Symbol('t')
x = [x1, x2]

f = np.array([x[1], (-p.g*sin(x[0]) - 2*p.L0*p.delta*x[1]**3)/(p.L0*(1+3*p.delta*x[0]*x[1]))])

# V = (Matrix(x + t * f).T @ Matrix(P) @ Matrix(x + t * f))
V = (Matrix(x + t * f).T @ Matrix(x + t * f))
W = integrate(V, (t, 0, d))


def gradient(scalar_function, variables):
    matrix_scalar_function = Matrix([scalar_function])
    return matrix_scalar_function.jacobian(variables)


gradW = gradient(W, x)
Wdot = gradW @ Matrix(f)
threshold = 0.2

size = 100
width = 2
x, y = np.meshgrid(np.linspace(-width, width, size), np.linspace(-width, width, size))
x1s = np.linspace(-2, 2, size)
x2s = np.linspace(-2, 2, size)

W_values = np.zeros(x.shape)
Wdot_values = np.zeros(x.shape)
W_indicator = np.zeros(x.shape)


for i in range(len(x1s)):
    x1v = x1s[i]
    print(f'i = {i}')
    for j in range(len(x2s)):
        x2v = x2s[j]

        Ws = W.subs(x1, x1v)
        Wss = Ws.subs(x2, x2v)
        W_values[i, j] = np.copy(Wss)

        Wdots = Wdot.subs(x1, x1v)
        Wdotss = Wdots.subs(x2, x2v)
        Wdot_values[i, j] = np.copy(Wdotss)

        if 0 >= float(Wdotss[0]):
            W_indicator[i, j] = -1
        else:
            W_indicator[i, j] = 1

fig = plt.figure(figsize=[12, 8])
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
mu = plt.contour(x, y, W_values, levels=20)
cbar = plt.colorbar(mu)

plt.xlabel(r'$\phi(t)$', size=20)
plt.ylabel(r'$\dot{\phi}(t)$', size=20, rotation=0)

plt.contourf(x, y, W_indicator, cmap='Reds', alpha=.3)

plt.ylim(-2, 2)
plt.xlim(-2, 2)
# plt.show()
plt.savefig('LF_FTLF.png', format='png', dpi=300)
plt.close(fig)

