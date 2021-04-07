import numpy as np
import matplotlib.pyplot as plt
from LF.NNLF.Functions import *
import dreal
import torch
import concise_pendulum
import pendulum
import torch.nn.functional as F
import timeit
import LF.NNLF.Functions as LNF


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # torch.nn.module will execute the __call__ method, which will automatically call the forward method defined in
        # your own class
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        # out = sigmoid(self.layer2(h_1))
        out = self.layer2(h_1) ** 2
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
D_in = 2             # input dimension
H1 = 15              # hidden dimension
D_out = 1            # output dimension
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
ball_lb = 0.25
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
    L_V = torch.diagonal(torch.mm(torch.mm(torch.mm(2*torch.sqrt(V_candidate), model.layer2.weight) \
                                           * dtanh(
        torch.tanh(torch.mm(x, model.layer1.weight.t()) + model.layer1.bias)), model.layer1.weight), f.t()), 0)

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
    b1 = model.layer1.bias.data.numpy()
    b2 = model.layer2.bias.data.numpy()

    # Falsification
    if i % 10 == 0:
        f = [x2,
             -(system.p.g * dreal.sin(x1) + 2 * system.p.L0 * system.p.delta * x2 ** 3) /
             (system.length(vars_) + 2 * system.p.L0 * system.p.delta * x1 * x2)]

        # Candidate V
        z1 = np.dot(vars_, w1.T) + b1

        a1 = []
        for j in range(0, len(z1)):
            a1.append(dreal.tanh(z1[j]))
        z2 = np.dot(a1, w2.T)+b2
        V_learn = (z2.item(0)) ** 2

        print('===========Verifying==========')
        start_ = timeit.default_timer()
        result= CheckLyapunov(vars_, f, V_learn, ball_lb, ball_ub, config, epsilon)
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
np.savetxt("b1.txt", model.layer1.bias.data, fmt="%s")
np.savetxt("b2.txt", model.layer2.bias.data, fmt="%s")

print('\n')
# print("Total time: ", stop - start)
# print("Verified time: ", t)

# out_iters += 1





