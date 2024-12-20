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

df = pd.DataFrame(columns=['N_e','K','simulation']) # in total z=0,159 inclusively per fold
z = 0 # total z=299
for K in [1,2,3,4,5,10]: 
    for N_e in [50,75,100,125,250]: # 25K baseline (50K total)
        for simulation in range(10):
                df.loc[z, 'N_e'] = N_e
                df.loc[z, 'K'] = K
                df.loc[z, 'simulation'] = simulation
                z += 1 

idx = 0 #int(os.environ["SLURM_ARRAY_TASK_ID"])
K = df.loc[idx,'K']
N_e = df.loc[idx,'N_e']
simulation = df.loc[idx, 'simulation']

N_i = N_e
N = N_e + N_i
D = int((K + N_e)/2)
sparsity = 0.25
U = 50
T = 200

J = np.random.normal(0, 1/np.sqrt(N), (N,N))
J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
J = J[:K,:]
J_pinv = np.linalg.pinv(J) # pseudo-inverse (J * J_inv = identity, but J_inv * J is not)

if K % 2 == 0:
    eigenvalues = np.load('eigenvalues_K_even_leq_10.npy')
else:
    eigenvalues = np.load('eigenvalues_K_odd_leq_5.npy')

eigenvalues = eigenvalues[:K]
trueA = generate_dynamics_A(eigenvalues) 


RNN = EI_subspace_RNN.EI_subspace_RNN(N_e, N_i, sparsity, J, seed=1)
zeta_alpha_beta_gamma_list = [(10**i,1,1,1/N) for i in list(np.arange(-3,0.5,0.5))] #[10 ** i for i in list(np.arange(-3,1,0.5))]
initW0, initW, loss_W, w_all = RNN.generate_or_initialize_weights_from_dynamics_LDS(A_target=trueA, R=0.85, zeta_alpha_beta_gamma_list = zeta_alpha_beta_gamma_list)
init_w = RNN.get_nonzero_weight_vector(initW)
initA = build_dynamics_matrix_A(initW, J)

true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R = RNN.generate_parameters(D, K)
true_x, true_y = RNN.generate_latents_and_observations(U, T, trueA, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R)

if simulation == 0: # initialize parameters from true ones
    max_iter = 20
    lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, alpha=U, beta=U, max_iter=max_iter)
else:
    max_iter = 100
    init_b, init_s, init_mu0, init_Q0, init_C_, init_d, init_R = RNN.generate_parameters(D, K)
    lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, alpha=U, beta=U, max_iter=max_iter)

np.savez(f'/models/N={N}_K={K}_simulation={simulation}_initialization_weights', initW0=initW0, initW =initW, loss_W = loss_W, w_all = w_all)
np.savez(f'/models/N={N}_K={K}_simulation={simulation}_fitting_EM', lossW=lossW, w=w, b=b, s=s, mu0=mu0, Q0=Q0, C_=C_, d=d, R=R)
np.savez(f'/models/N={N}_K={K}_simulation={simulation}_true_parameters_and_data', J=J, true_x=true_x, true_y=true_y, trueA=trueA, true_b=true_b, true_s=true_s, true_mu0=true_mu0, true_Q0=true_Q0, true_C_=true_C_, true_d=true_d, true_R=true_R)

