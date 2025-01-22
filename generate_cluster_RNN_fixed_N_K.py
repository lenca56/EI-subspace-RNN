import numpy as np
import pandas as pd
import EI_subspace_RNN
import scipy.stats as stats
import scipy.linalg
from utils import *
from plotting_utils import *
import matplotlib.pyplot as plt

import sys
import os

df = pd.DataFrame(columns=['K','simulation']) # in total z=0,239 
z = 0 
for K in [1,2,3,4,5,10,25,48]:
    for simulation in range(30):
        df.loc[z, 'K'] = K
        df.loc[z, 'simulation'] = simulation
        z += 1 

idx = 0 #int(os.environ["SLURM_ARRAY_TASK_ID"])
K = df.loc[idx, 'K']
simulation = df.loc[idx, 'simulation']

N_e = 100
N_i = N_e
N = N_e + N_i
D = 50
sparsity = 0.25
U = 200
T = 100

J_possibilities = []

# Case 1 - normal J
J = np.random.normal(0, 1/np.sqrt(N), (N,N))
J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
J = J[:K,:]
J_possibilities.append(J)

# Case 2 - (Kth dimension is co-activation pattern)
J = np.random.normal(0, 1/np.sqrt(N), (N,N))
J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
if K > 1:
    J = J[:K-1,:]
    J[K-1,:] = 1/np.sqrt(N)
else:
    J[0,:] = 1/np.sqrt(N)
J_possibilities.append(J)

# Case 3 - uniform J
J = np.random.uniform(0, 1, (N,N))
J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
J = J[:K,:] / 1/np.sqrt(N)
J_possibilities.append(J)

eigenvalues = generate_eigenvalues(K)
trueA = generate_dynamics_A(eigenvalues) 

for i in range(len(J_possibilities)):
    J = J_possibilities[i]
    J_pinv = np.linalg.pinv(J) # pseudo-inverse (J * J_inv = identity, but J_inv * J is not)

    RNN = EI_subspace_RNN.EI_subspace_RNN(N_e, N_i, sparsity, J, seed=1)
    zeta_alpha_beta_gamma_list = [(10**i,1,1,0) for i in list(np.arange(-1,0.5,0.5))]
    initW0, initW, loss_W, w_all = RNN.generate_or_initialize_weights_from_dynamics_LDS(A_target=trueA, R=0.85, zeta_alpha_beta_gamma_list = zeta_alpha_beta_gamma_list)
    init_w = RNN.get_nonzero_weight_vector(initW)
    initA = build_dynamics_matrix_A(initW, J)

    true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R = RNN.generate_parameters(D, K)
    true_x, true_y = RNN.generate_latents_and_observations(U, T, trueA, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R)
    # lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, alpha=10, beta=10, max_iter=10)

    # np.savez(f'models/N={N}_K={K}_parameters_J_possbility_{i}', initW0=initW0, initW =initW, loss_W = loss_W, w_all = w_all, lossW=lossW, w=w, b=b, s=s, mu0=mu0, Q0=Q0, C_=C_, d=d, R=R, J=J, true_x=true_x, true_y=true_y, trueA=trueA, true_b=true_b, true_s=true_s, true_mu0=true_mu0, true_Q0=true_Q0, true_C_=true_C_, true_d=true_d, true_R=true_R)
    

# if simulation == 0: # initialize parameters from true ones
#     max_iter = 20
#     lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, alpha=10, beta=10, max_iter=max_iter)
# else:
#     max_iter = 200
#     init_b, init_s, init_mu0, init_Q0, init_C_, init_d, init_R = RNN.generate_parameters(D, K)
#     lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, init_b, init_s, init_mu0, init_Q0, init_C_, init_d, init_R, alpha=10, beta=10, max_iter=max_iter)

