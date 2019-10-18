import numpy as np


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def last_consecutives(vals, step=1):
    '''
    Find the last consecutive group of numbers
    :param vals:
    :param step:
    :return:
    '''
    group = []
    expected = None
    for val in reversed(vals):
        if (val == expected) or (expected is None):
            group.append(val)
        else:
            break
        expected = val - step
    return group

