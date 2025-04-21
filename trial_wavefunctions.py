
import numpy as np
import matplotlib.pyplot as plt


def helium_1_param(r1 ,r2, alpha, batch=False):
    '''Computes the trial wavefunction'''

    if batch:
        axis = 1
    else:
        axis = 0

    norm_r1 = np.linalg.norm(r1, axis=axis)
    norm_r2 = np.linalg.norm(r2, axis=axis)
    r12 = np.linalg.norm(r1-r2, axis=axis)
    wf = np.exp(-2 * norm_r1) * np.exp(-2 * norm_r2) * np.exp(r12 / (2 * (1 + alpha * r12)))
    return wf

def helium_1_param_vec(r1 ,r2, alpha):
    '''Computes the trial wavefunction'''

    norm_r1 = np.linalg.norm(r1, axis=1)
    norm_r2 = np.linalg.norm(r2, axis=1)
    r12 = np.linalg.norm(r1-r2, axis=1)
    wf = np.exp(-2 * norm_r1) * np.exp(-2 * norm_r2) * np.exp(r12 / (2 * (1 + alpha * r12)))
    return wf