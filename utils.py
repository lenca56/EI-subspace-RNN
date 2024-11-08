import numpy as np
import EI_subspace_RNN
import scipy.stats as stats

def build_dynamics_matrix_A(W, J):
    return J @ W @ np.linalg.pinv(J)

