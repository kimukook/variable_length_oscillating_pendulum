# # -*- coding: utf-8 -*-
# import dreal
# import torch
# import numpy as np
# import random
#
#
# def CheckLyapunov(x, f, V, ball_lb, ball_ub, config, epsilon):
#     # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
#     # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub).
#     # If it return unsat, then there is no state violating the conditions.
#
#     ball= dreal.Expression(0)
#     lie_derivative_of_V = dreal.Expression(0)
#
#     for i in range(len(x)):
#         ball += x[i]*x[i]
#         lie_derivative_of_V += f[i]*V.Differentiate(x[i])
#     ball_in_bound = dreal.logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
#
#     # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)
#     condition = dreal.logical_and(dreal.logical_imply(ball_in_bound, V >= 0),
#                            dreal.logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon))
#     return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
#
#
# def AddCounterexamples(x, CE, N):
#     new_sample = torch.cat((torch.Tensor(N, 1).uniform_(CE[0].lb(), CE[0].ub()), torch.Tensor(N, 1).uniform_(CE[1].lb(), CE[1].ub())), 1)
#     x = torch.cat((x, new_sample), 0)
#     return x
#
#
# def dtanh(s):
#     # Derivative of activation
#     return 1.0 - s**2
#
#
# def dsquare(s):
#     return 2 * np.sqrt(s)
#
#
# def Tune(x):
#     # Circle function values
#     y = []
#     for r in range(0, len(x)):
#         v = 0
#         for j in range(x.shape[1]):
#             v += x[r][j]**2
#         f = [torch.sqrt(v)]
#         y.append(f)
#     y = torch.tensor(y)
#     return y
from dreal import *
import torch
import numpy as np
import random


def CheckLyapunov(x, f, V, ball_lb, ball_ub, config, epsilon):
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub).
    # If it return unsat, then there is no state violating the conditions.

    ball = Expression(0)
    lie_derivative_of_V = Expression(0)

    for i in range(len(x)):
        ball += x[i] * x[i]
        lie_derivative_of_V += f[i] * V.Differentiate(x[i])
    ball_in_bound = logical_and(ball_lb * ball_lb <= ball, ball <= ball_ub * ball_ub)

    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)
    condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                            logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon))
    return CheckSatisfiability(logical_not(condition), config)


def AddCounterexamples(x, CE, N):
    # Adding CE back to sample set
    c = []
    nearby = []
    for i in range(CE.size()):
        c.append(CE[i].mid())
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.random.uniform(lb, ub, N)
        nearby.append(nearby_)
    for i in range(N):
        n_pt = []
        for j in range(x.shape[1]):
            n_pt.append(nearby[j][i])
        x = torch.cat((x, torch.tensor([n_pt])), 0)
    return x


def dtanh(s):
    # Derivative of activation
    return 1.0 - s ** 2


def Tune(x):
    # Circle function values
    y = []
    for r in range(0, len(x)):
        v = 0
        for j in range(x.shape[1]):
            v += x[r][j] ** 2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y)
    return y
