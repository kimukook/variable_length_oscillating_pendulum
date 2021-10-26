import numpy as np
import torch
import torch.nn.functional as F
import concise_pendulum
import itertools

'''
This is a script that customizes the neural network to suits new derivation of NNLF for proving the asymptotic stability 
of the Variable-Length Pendulum (VLP) problem. 

=====================================
Author  :  Muhan Zhao
Date    :  Oct. 22, 2021
Location:  UC San Diego, La Jolla, CA
=====================================
'''


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        # G11, G12, G2, G3 are parameters to be optimized
        # TODO how to initialize the parameters? Temporarily using xavier uniform
        # if you set all Gs to be zeros, the Vdot calculation gives you all zeros, fail to give labels.
        self.G11 = torch.nn.Parameter(torch.Tensor(n_hidden, n_input))
        self.G12 = torch.nn.Parameter(torch.Tensor(n_hidden - n_input, n_input))
        self.G2 = torch.nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.G3 = torch.nn.Parameter(torch.Tensor(n_output, n_hidden))

        self.G11.data = torch.nn.init.xavier_uniform_(self.G11.data)
        self.G12.data = torch.nn.init.xavier_uniform_(self.G12.data)
        self.G2.data = torch.nn.init.xavier_uniform_(self.G2.data)
        self.G3.data = torch.nn.init.xavier_uniform_(self.G3.data)

        self.epsilon = 1e-2

    def forward(self, x):
        # torch.nn.module will execute the __call__ method, which will automatically call the forward method defined in
        # your own class
        sigmoid = torch.nn.Tanh()
        W1 = torch.cat((self.G11.T @ self.G11 + self.epsilon * torch.eye(self.n_input), self.G12), 0)
        h1 = sigmoid(W1 @ x)
        h2 = sigmoid((self.G2.T @ self.G2 + self.epsilon * torch.eye(self.n_hidden)) @ h1)
        h3 = sigmoid((self.G3.T @ self.G3 + self.epsilon * torch.eye(self.n_output)) @ h2)
        out = torch.sum(h3 * h3, 0).reshape(-1, 1)
        return out


'''
===========================================
Set up the dynamic system
===========================================
'''
m = 1
g = 9.8
L0 = 1
delta = 0.05
Lmax = 1.5
Lmin = 0.5
params = concise_pendulum.PendulumParams(m, g, L0, delta, Lmax, Lmin)
system = concise_pendulum.Pendulum(params)


def dynamic(x):
    '''
    This function outputs the system dynamic given the states in x.
    :param x:
    :return:
    '''
    # x should be 2-by-SampleSize vector
    f = []
    for i in range(x.shape[1]):
        f_x = [x[1, i],
               -(system.p.g * np.sin(x[0, i]) + 2 * system.p.L0 * system.p.delta * x[1, i]**3) /
               (system.length(x[:, i]) + 2 * system.p.L0 * system.p.delta * x[0, i] * x[1, i])]
        f.append(f_x)
    f = torch.tensor(f)
    return f.T


def rk4(x, dt, func):
    '''
    Runge-kutta 4 time marching scheme
    :param x:
    :param dt:
    :param func:
    :return:
    '''
    f1 = func(x)
    f2 = func(x + dt / 2 * f1)
    f3 = func(x + dt / 2 * f2)
    f4 = func(x + dt * f3)
    x_new = x + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    return x_new


def label_creator(x, V, model, dynamic):
    '''
    This function outputs the label y=+1/-1 for each point given in x. The label is decided by the rule if x is in the
    safe region and Vdot(x) is negative. Instead of directly compute Vdot via mathematical deduction, here we use
                        V(f(x)) - V(x) to verify the decrease condition of Lyapunov function along the trajectory.
    The advantage of deriving Vdot in this way is that, you dont have to recompute the Vdot(x) everytime you change the
    Neural Network model.

    The disadvantage is, this approximation might not exactly recover the true Vdot for each point of x.

    :param x        :   states of interest
    :param V        :   Lyapunov function values at x
    :param model    :   neural network model for evaluating V
    :param dynamic  :   dynamic of the system of interest
    :return:
    '''
    dt = .01
    x_next = rk4(x, dt, dynamic)
    V_next = model(x_next)
    dV = V_next - V
    labels = torch.sign(dV)
    return labels, dV


def maximum_level_set(model, lb, ub, size=40):
    '''

    :param model:
    :param bnds:    example : bnds = np.hstack((np.ones((2, 1)), -np.ones((2, 1))))
    :return:
    '''
    # First, generate the discretized grid points on the boundary of region of interest
    n = lb.shape[0]
    x = np.empty(shape=[n, 0])

    # generate corner points
    bnds = np.kron(np.ones((1, 2 ** n)), ub)
    for ii in range(n):
        tt = np.mod(np.arange(2 ** n) + 1, 2 ** (n - ii)) <= 2 ** (n - ii - 1) - 1
        bnds[ii, tt] = lb[ii]

    # generate grid points by visiting all corner points
    for i in range(bnds.shape[1]):
        if i < bnds.shape[1] - 1:
            x = np.hstack((x, np.))




    return
'''
===========================================
Set up the parameters for learning
===========================================
'''
N = 2000             # sample size
D_in = 2            # input dimension
D_h = 6              # hidden dimension
D_out = 1           # output dimension
torch.manual_seed(10)

x = torch.Tensor(D_in, N).uniform_(-1, 1)
x_0 = torch.zeros([1, 2])

# set up the model
model = Net(D_in, D_h, D_out)
max_iter = 1000
valid = False

# set up the parameter for the cost function
lagrangian = 1

# set up the optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
===========================================
The learning phase
===========================================
'''

for i in itertools.count():
    if i >= max_iter or valid:
        break
    else:
        V_candidate = model(x)
        f = dynamic(x)

        # First, identify which points in x that satisfy equation (5), i.e. create the labels y=+1/-1
        y, dV = label_creator(x, V_candidate, model, dynamic)

        L_cost =
        # Since I have already estimate the region of attraction, I skip one step here, that is, sample points from the
        # safe region, using the points from V_theta(alpha * c_k) to T-steps simulate points forward, if points stay in
        # V_theta(ck), we know its trajectory is gonna converge, then those points can be added to safe region to form a
        # new estimate of the safe region, but in this work we know that all those points are classified as safe, notice
        # that I have the precomputed safe region. so I am cheating to omit this part now.

        # Second, generate the reward/cost function using equation (7) from Felix, optimize it using ADAM

        # third, iterate


