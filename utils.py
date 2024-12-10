import numpy as np
import EI_subspace_RNN
import scipy.stats as stats
import scipy.linalg

def generate_dynamics_A(eigenvalues):
    '''
    eigenvectors: np array
        columns are eigenvectors
    '''
    comp_A = scipy.linalg.companion(np.poly((eigenvalues))) # companion matrix from characteristic polynomial
    K = eigenvalues.shape[0]
    # generate real random matrix for similarity transformation of companion matrix for given eigenvalues
    P = np.random.rand(K,K) # uniform (0,1)
    trueA = np.linalg.inv(P)  @ comp_A @ P # similarity
    return trueA

def build_dynamics_matrix_A(W, J):
    return J @ W @ np.linalg.pinv(J)

def mse(z, true_z):
    return np.trace((z-true_z) @ (z-true_z).T)

def angle_vectors(v1, v2):
    # potentially complex vectors v1 and v2
    cos_angle = np.real(np.vdot(v1,v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    return np.rad2deg(angle_rad)

