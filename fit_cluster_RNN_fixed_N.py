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

df = pd.DataFrame(columns=['K','simulation','ei']) # in total z=0,299 
z = 0 
for K in [1,2,3,5]:#[1,2,3,5,15,25]:
    for ei in [0,1,2,3]:
        for simulation in range(30):
            df.loc[z, 'K'] = K
            df.loc[z, 'ei'] = ei
            df.loc[z, 'simulation'] = simulation
            z += 1 

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
K = df.loc[idx, 'K']
ei = df.loc[idx, 'ei']
simulation = df.loc[idx, 'simulation']

N_e = 100
N_i = N_e
N = N_e + N_i
D = 30
sparsity = 0.25
U = 500
T = 200
zeta_alpha_beta_gamma_list = [(10**i,1,1,10**(i-2.5)) for i in list(np.arange(-1.5,0.5,0.25))]
J_possibilities = []

# # Case 0 - normal J with orthogonality imposed (QR)
# J = np.random.normal(0, 1/np.sqrt(N), (N,N))
# J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
# J = J[:K,:]
# J_possibilities.append(J)

# Case 1 - normal J with no orthogonality imposed
J = np.random.normal(0, 1/np.sqrt(N), (N,N))
J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
J = J[:K,:]
J_possibilities.append(J)

# Case 2 - J normal but (Kth dimension is co-activation pattern)
J = np.random.normal(0, 1/np.sqrt(N), (N,N))
J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
if K > 1:
    J = J[:K,:]
    J[K-1,:] = 1/np.sqrt(N)
else:
    J = J[0,:].reshape((1,N))
    J[0,:] = 1/np.sqrt(N)
J_possibilities.append(J)

# Case 3 - uniform J
J = np.random.uniform(0, 1, (N,N))
J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
J = J[:K,:] / 1/np.sqrt(N)
J_possibilities.append(J)

# generate dynamics (either normal or non-normal)
eigenvalues = generate_eigenvalues(K)
if simulation < 10:
    trueA = generate_dynamics_A(eigenvalues, normal=True) 
else:
    trueA = generate_dynamics_A(eigenvalues, normal=False) 

for i in range(len(J_possibilities)):
    J = J_possibilities[i]
    # pseudo-inverse (J * J_inv = identity, but J_inv * J is not)
    J_pinv = np.linalg.pinv(J) 
    RNN = EI_subspace_RNN.EI_subspace_RNN(N_e, N_i, sparsity, J, seed=1)
    initW0, initW, loss_W, w_all = RNN.generate_or_initialize_weights_from_dynamics_LDS(A_target=trueA, R=0.85, zeta_alpha_beta_gamma_list = zeta_alpha_beta_gamma_list)
    init_w = RNN.get_nonzero_weight_vector(initW)
    initA = build_dynamics_matrix_A(initW, J)

    true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R = RNN.generate_parameters(D, K)
    true_x, true_y = RNN.generate_latents_and_observations(U, T, trueA, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R)
    if ei == 0:
        ecll, ll, lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, alpha=10, beta=10, max_iter=25)
    elif ei == 1:
        ecll, ll, lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, alpha=10, beta=0, max_iter=25)
    elif ei == 2:
        ecll, ll, lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, alpha=0, beta=10, max_iter=25)
    else:
        ecll, ll, lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, alpha=0, beta=0, max_iter=25)
    
    fitW = RNN.build_full_weight_matrix(w)
    unstable, n_unstable = check_unstable(fitW)
    if unstable == False: # stable 
        v = RNN.generate_network_activity(U, T, fitW, b, s, mu0, Q0)
        # print('generated network activity')
    else:
        v = np.zeros((1))

    np.savez(f'models/N={N}_K={K}_parameters_EI={ei}_simulation_{simulation}_J_possibility_{i}', ecll=ecll, ll=ll, initW0=initW0, initW =initW, loss_W = loss_W, w_all = w_all, lossW=lossW, fitW=fitW, b=b, s=s, mu0=mu0, Q0=Q0, C_=C_, d=d, R=R, J=J, true_x=true_x, true_y=true_y, trueA=trueA, true_b=true_b, true_s=true_s, true_mu0=true_mu0, true_Q0=true_Q0, true_C_=true_C_, true_d=true_d, true_R=true_R, v=v)
    