import matplotlib.pyplot as plt
from utils import *
import numpy as np
import EI_subspace_RNN

def plot_mse_parameters(axes, A, b, s, mu0, Q0, C_, d, R, true_A, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R):
    axes.set_ylabel('mse')
    # axes.set_ylim(0,0.001)
    axes.scatter(range(9), [mse(A, true_A), mse(b[0], true_b[0]), mse(b[1], true_b[1]), (s - true_s) ** 2, mse(mu0, true_mu0), mse(Q0, true_Q0), mse(C_, true_C_), mse(d, true_d), mse(R, true_R)])
    axes.set_xticks(range(9), ['A', 'b0', 'b1','s', 'mu0', 'Q0', 'C_', 'd', 'R'])
    plt.show()

def plot_eigenvalues(axes, eigval, color='black', label=''):
    axes.scatter(np.real(eigval), np.imag(eigval), color=color, label=label)
    axes.set_xlabel('Re(eigenvalue)')
    axes.set_ylabel('Im(eigenvalue)')
    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    axes.add_patch(circle1)
    axes.axvline(0, linestyle='dashed', color='black')
    axes.axhline(0, linestyle='dashed', color='black')