import numpy as np
import matplotlib.pyplot as plt
from LF.NNLF.Functions import *
import dreal
import torch
import concise_pendulum
import torch.nn.functional as F
import timeit
import LF.NNLF.Functions as LNF
from matplotlib import cm


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer4 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer5 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # torch.nn.module will execute the __call__ method, which will automatically call the forward method defined in
        # your own class
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        h_3 = sigmoid(self.layer3(h_2))
        h_4 = sigmoid(self.layer4(h_3))
        out = sigmoid(self.layer5(h_4))
        return out


'''
set up dynamical system
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
    # x should be samplesize-by-2 vector
    f = []
    for i in range(x.shape[0]):
        f_x = [x[i, 1],
               -(system.p.g * np.sin(x[i, 0]) + 2 * system.p.L0 * system.p.delta * x[i, 1]**3) /
               (system.length(x[i, :]) + 2 * system.p.L0 * system.p.delta * x[i, 0] * x[i, 1])]
        f.append(f_x)
    f = torch.tensor(f)
    return f


'''
For learning 
'''
N = 2000             # sample size
D_in = 2            # input dimension
H1 = 6              # hidden dimension
D_out = 1           # output dimension
torch.manual_seed(10)

x = torch.Tensor(N, D_in).uniform_(-1, 1)
x_0 = torch.zeros([1, 2])

'''
For verifying 
'''
x1 = dreal.Variable("x1")
x2 = dreal.Variable("x2")
vars_ = [x1, x2]

config = dreal.Config()
config.use_polytope_in_forall = True
config.use_local_optimization = True
config.precision = 1e-2
epsilon = 0
# Checking candidate V within a ball around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub)
ball_lb = 0.5
ball_ub = 1


# Learning and Falsification

# out_iters = 0
valid = False
# while out_iters < 2 and not valid:
#     start = timeit.default_timer()

model = Net(D_in, H1, D_out)
L = []
i = 0
max_iters = 2000
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

while i < max_iters and not valid:
    i += 1

    V_candidate = model(x)
    V_x0 = model(x_0)
    f = dynamic(x)

    # Functions.Tune
    Circle_Tuning = Tune(x)
    # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
    # L_V = torch.diagonal(torch.mm(torch.mm(torch.mm(dtanh(V_candidate), model.layer2.weight) \
    #                                        * dtanh(
    #     torch.tanh(torch.mm(x, model.layer1.weight.t()) + model.layer1.bias)), model.layer1.weight), f.t()), 0)

    # L_V = torch.diagonal(dtanh(V_candidate) @ model.layer3.weight * (dtanh(torch.tanh(x @ model.layer1.weight.t() + model.layer1.bias) @ model.layer2.weight.t() + model.layer2.bias) @ model.layer2.weight) * dtanh(x @ model.layer1.weight.t() + model.layer1.bias) @ model.layer1.weight @ f.t(), 0)
    L_V = torch.sum(dtanh(V_candidate) @ model.layer5.weight *
                    dtanh(torch.tanh(torch.tanh(torch.tanh(torch.tanh(x@ model.layer1.weight.t() + model.layer1.bias) @ model.layer2.weight.t() + model.layer2.bias) @ model.layer3.weight.t() + model.layer3.bias) @ model.layer4.weight.t() + model.layer4.bias)) @ model.layer4.weight *
                    dtanh(torch.tanh(torch.tanh(torch.tanh(x @ model.layer1.weight.t() + model.layer1.bias) @ model.layer2.weight.t() + model.layer2.bias) @ model.layer3.weight.t() + model.layer3.bias)) @ model.layer3.weight *
                    dtanh(torch.tanh(torch.tanh(x @ model.layer1.weight.t() + model.layer1.bias) @ model.layer2.weight.t() + model.layer2.bias)) @ model.layer2.weight *
                    dtanh(torch.tanh(x @ model.layer1.weight.t() + model.layer1.bias)) @ model.layer1.weight * f, 1)

    # f.t() -> f transpose
    # With tuning term
    Lyapunov_risk = (F.relu(-V_candidate) + 1.5 * F.relu(L_V + 0.5)).mean() \
                    + 2.2 * ((Circle_Tuning - 6 * V_candidate).pow(2)).mean() + V_x0.pow(2)
    # Lyapunov_risk = (F.relu(-V_candidate) + 1.5 * F.relu(L_V + 0.5)).mean() \
    #                 + 2.2 * (Circle_Tuning - 6 * V_candidate).mean() + V_x0.pow(2)

    # Without tuning term
    # Lyapunov_risk = (F.relu(-V_candidate) + 1.5*F.relu(L_V+0.5)).mean() + 1.2*V_x0.pow(2)

    print(i, "Lyapunov Risk =", Lyapunov_risk.item())

    L.append(Lyapunov_risk.item())
    optimizer.zero_grad()
    Lyapunov_risk.backward()
    optimizer.step()

    w1 = model.layer1.weight.data.numpy()
    w2 = model.layer2.weight.data.numpy()
    w3 = model.layer3.weight.data.numpy()
    w4 = model.layer4.weight.data.numpy()
    w5 = model.layer5.weight.data.numpy()

    b1 = model.layer1.bias.data.numpy()
    b2 = model.layer2.bias.data.numpy()
    b3 = model.layer3.bias.data.numpy()
    b4 = model.layer4.bias.data.numpy()
    b5 = model.layer5.bias.data.numpy()

    # Falsification
    if i % 10 == 0:
        f = [x2,
             -(system.p.g * dreal.sin(x1) + 2 * system.p.L0 * system.p.delta * x2 ** 3) /
             (system.length(vars_) + 2 * system.p.L0 * system.p.delta * x1 * x2)]

        z1 = np.dot(vars_, w1.T) + b1
        a1 = [dreal.tanh(z) for z in z1]
        z2 = np.dot(a1, w2.T) + b2
        a2 = [dreal.tanh(z) for z in z2]
        z3 = np.dot(a2, w3.T) + b3
        a3 = [dreal.tanh(z) for z in z3]
        z4 = np.dot(a3, w4.T) + b4
        a4 = [dreal.tanh(z) for z in z4]
        z5 = np.dot(a4, w5.T) + b5
        V_learn = dreal.tanh(z5.item(0))

        print('===========Verifying==========')
        start_ = timeit.default_timer()
        result = CheckLyapunov(vars_, f, V_learn, ball_lb, ball_ub, config, epsilon)
        stop_ = timeit.default_timer()

        if result:
            print("Not a Lyapunov function. Found counterexample: ")
            print(result)
            x = LNF.AddCounterexamples(x, result, 10)
        else:
            valid = True
            print("Satisfy conditions!!")
            print(V_learn, " is a Lyapunov function.")
        print('==============================')


stop = timeit.default_timer()

np.savetxt("w1.txt", model.layer1.weight.data, fmt="%s")
np.savetxt("w2.txt", model.layer2.weight.data, fmt="%s")
np.savetxt("w3.txt", model.layer3.weight.data, fmt="%s")
np.savetxt("w4.txt", model.layer4.weight.data, fmt="%s")
np.savetxt("w5.txt", model.layer5.weight.data, fmt="%s")

np.savetxt("b1.txt", model.layer1.bias.data, fmt="%s")
np.savetxt("b2.txt", model.layer2.bias.data, fmt="%s")
np.savetxt("b3.txt", model.layer3.bias.data, fmt="%s")
np.savetxt("b4.txt", model.layer4.bias.data, fmt="%s")
np.savetxt("b5.txt", model.layer5.bias.data, fmt="%s")

print('\n')


#### plotting


def nnlf_tanh_eval(x):
    hidden1 = w1 @ x + b1.reshape(-1, 1)
    hidden2 = w2 @ np.tanh(hidden1) + b2.reshape(-1, 1)
    hidden3 = w3 @ np.tanh(hidden2) + b3.reshape(-1, 1)
    output = w4 @ np.tanh(hidden3) + b4.reshape(-1, 1)
    return np.tanh(output)


def f_eval(x):
    f = np.zeros((2, 1))
    f[0] = x[1]
    f[1] = -(system.p.g * np.sin(x[0]) + 2 * system.p.L0 * system.p.delta * x[1] ** 3) / \
           (system.length(x) + 2 * system.p.L0 * system.p.delta * x[0] * x[1])
    return f


def plot(r, step_size):
    x, y = np.linspace(-r, r, step_size), np.linspace(-r, r, step_size)
    X, Y = np.meshgrid(x, y)

    V = np.zeros(X.shape)
    Vdot = np.zeros(X.shape)

    for i in range(step_size):
        for j in range(step_size):
            point = np.array([X[i, j], Y[i, j]]).reshape(-1, 1)
            V[i, j] = nnlf_tanh_eval(point)
            Vdot[i, j] = dtanh(V[i, j]) * w4 * \
                         dtanh(np.tanh(w3 @ np.tanh(w2 @ np.tanh(w1 @ point + b1.reshape(-1, 1)) + b2.reshape(-1, 1)) + b3.reshape(-1, 1))).T @ w3 * \
                         dtanh(np.tanh(w2 @ np.tanh(w1 @ point + b1.reshape(-1, 1)) + b2.reshape(-1, 1))).T @ w2 * \
                         dtanh(np.tanh(w1 @ point + b1.reshape(-1, 1)).T) @ w1 @ f_eval(point)

    # plot V
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.contour(X, Y, Vdot, zdir='z', offset=0, cmap=cm.coolwarm)
    ax.plot_surface(X, Y, V, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
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
    ax.contour(X, Y, Vdot, zdir='z', offset=-2, cmap=cm.coolwarm)
    ax.plot_surface(X, Y, Vdot, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
    plt.xlabel(r'$\theta$', fontweight='bold', position=(0.5, 1))
    plt.ylabel(r'$\dot{\theta}$', fontweight='bold', position=(0.5, 0))
    ax.set_zlabel('$\dot{V}$')
    ax.axes.set_zlim3d(bottom=-2, top=.5)

    plt.title('The Lie derivative of Lyapunov Function')
    # plt.show()
    plt.savefig('NNLF_Vdot.png', format='png', dpi=300)

    ax.view_init(elev=0, azim=0)
    ax.set_xticks([])
    plt.savefig('NNLF_Vdot2.png', format='png', dpi=300)
    plt.close(fig)

    # Vdot from x axis
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.contour(X, Y, Vdot, zdir='z', offset=-2, cmap=cm.coolwarm)
    ax.plot_surface(X, Y, Vdot, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
    plt.xlabel(r'$\theta$', fontweight='bold', position=(0.5, 1))
    plt.ylabel(r'$\dot{\theta}$', fontweight='bold', position=(0.5, 0))
    ax.set_zlabel('$\dot{V}$')
    ax.axes.set_zlim3d(bottom=-2, top=.5)

    ax.view_init(elev=0, azim=90)
    ax.set_yticks([])
    plt.savefig('NNLF_Vdot3.png', format='png', dpi=300)
    plt.close(fig)


r = 1
step_size = 100
plot(r, step_size)

