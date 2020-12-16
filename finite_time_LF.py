import pendulum
from pendulum import Pendulum
import numpy as np


'''
This is a script that finds the finite-time Lyapunov Function. 
And construct the Lyapunov function.
=====================================
Author  :  Muhan Zhao
Date    :  Dec. 15, 2019
Location:  UC San Diego, La Jolla, CA
=====================================
'''


class FTLF:
    def __init__(self, attributes):
        self.d = attributes.get('finite-time d', 0.2)
        self.dt = None

    @staticmethod
    def compute_lf(self, vary_length_pendulum: Pendulum):
        self.dt = vary_length_pendulum.dt
        steps = round(self.d / self.dt)
        if steps > vary_length_pendulum.steps:
            raise ValueError('FTLF time length > pendulum simulated time steps! Reduce d.')
        else:
            pass
        assemble_states = np.vstack((vary_length_pendulum.asym_control_phi, vary_length_pendulum.asym_control_dphi))
        w = np.sum(np.linalg.norm(assemble_states, axis=0))
        return w


if __name__ == '__main__':
    d = .2
    properties = {
        'finite-time d': d,
    }

