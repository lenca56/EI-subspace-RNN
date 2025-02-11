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

df = pd.DataFrame(columns=['K','simulation']) # in total z=0,299 
z = 0
for K in [1,2,3,5,10]:
    for simulation in range(60):
        df.loc[z, 'K'] = K
        # df.loc[z, 'ei'] = ei
        df.loc[z, 'simulation'] = simulation
        z += 1 

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
K = df.loc[idx, 'K']
# ei = df.loc[idx, 'ei']
simulation = df.loc[idx, 'simulation']

N_e = 100
N_i = N_e
N = N_e + N_i
D = 20
sparsity = 0.25
U = 250
T = 100
max_iter = 10

# J_possibilities = []

# # Case 1 - normal J with no orthogonality imposed
# J = np.random.normal(0, 1/np.sqrt(N), (N,N))
# # J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
# J = J[:K,:]
# J_possibilities.append(J)

# # Case 2 - J normal but (Kth dimension is co-activation pattern)
# J = np.random.normal(0, 1/np.sqrt(N), (N,N))
# # J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
# if K > 1:
#     J = J[:K,:]
#     J[K-1,:] = 1/np.sqrt(N)
# else:
#     J = J[0,:].reshape((1,N))
#     J[0,:] = 1/np.sqrt(N)
# J_possibilities.append(J)

# # Case 3 - uniform J
# J = np.random.uniform(0, 1, (N,N))
# # J, _ = np.linalg.qr(J)  # QR decomposition, Q is the orthogonal matrix
# J = J[:K,:] / np.sqrt(N)
# J_possibilities.append(J)

# # generate dynamics (either normal or non-normal)
# eigenvalues = generate_eigenvalues(K)
# if simulation < 10:
#     trueA = generate_dynamics_A(eigenvalues, normal=True) 
# else:
#     trueA = generate_dynamics_A(eigenvalues, normal=False) 

trueA = np.load(f'models/N={N}_K={K}/EI=0_simulation_{simulation}_J_possibility_0.npz', allow_pickle=True)['trueA']

for ei_control in [0,1,2,3]:

    # J = J_possibilities[i]
    J = np.load(f'models/N={N}_K={K}/EI=0_simulation_{simulation}_J_possibility_{0}.npz', allow_pickle=True)['J']
    RNN = EI_subspace_RNN.EI_subspace_RNN(N_e, N_i, sparsity, J, seed=1)

    if ei_control == 0: # no Dale's law and usual constraints
        Dale_law = 0
        zeta_alpha_beta_gamma_list = [(10**i,1,1,10**(i-2.5)) for i in list(np.arange(-1,0.5,0.25))]
        initW0, initW, loss_W, w_all = RNN.generate_or_initialize_weights_from_dynamics_LDS(A_target=trueA, R=0.85, zeta_alpha_beta_gamma_list = zeta_alpha_beta_gamma_list, Dale_law=Dale_law)
        init_w = RNN.get_nonzero_weight_vector(initW, Dale_law=Dale_law)
        initA = build_dynamics_matrix_A(initW, J)
        true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R = RNN.generate_parameters(D, K)
        true_x, true_y = RNN.generate_latents_and_observations(U, T, trueA, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R)
        ecll, ll, lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, Dale_law=Dale_law, alpha=10, beta=10, max_iter=max_iter)
        fitW = RNN.build_full_weight_matrix(w, Dale_law=Dale_law)
        unstable, n_unstable = check_unstable(fitW)
        if unstable == False: # stable 
            v = RNN.generate_network_activity(U, T, fitW, b, s, mu0, Q0)
            # print('generated network activity')
        else:
            v = np.zeros((1))
        np.savez(f'models/N={N}_K={K}/EI_control={ei_control}_simulation_{simulation}_J_possibility_{0}', ecll=ecll, ll=ll, initW0=initW0, initW =initW, loss_W = loss_W, w_all = w_all, lossW=lossW, fitW=fitW, b=b, s=s, mu0=mu0, Q0=Q0, C_=C_, d=d, R=R, J=J, true_x=true_x, true_y=true_y, trueA=trueA, true_b=true_b, true_s=true_s, true_mu0=true_mu0, true_Q0=true_Q0, true_C_=true_C_, true_d=true_d, true_R=true_R, v=v)
        
    elif ei_control == 1: # no Dale's law and only self-containment
        Dale_law = 0
        zeta_alpha_beta_gamma_list = [(10**i,1,0,10**(i-2.5)) for i in list(np.arange(-1,0.5,0.25))]
        initW0, initW, loss_W, w_all = RNN.generate_or_initialize_weights_from_dynamics_LDS(A_target=trueA, R=0.85, zeta_alpha_beta_gamma_list = zeta_alpha_beta_gamma_list, Dale_law=Dale_law)
        init_w = RNN.get_nonzero_weight_vector(initW, Dale_law=Dale_law)
        initA = build_dynamics_matrix_A(initW, J)
        true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R = RNN.generate_parameters(D, K)
        true_x, true_y = RNN.generate_latents_and_observations(U, T, trueA, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R)
        ecll, ll, lossW, w, b, s, mu0, Q0, C_, d, R = RNN.fit_EM(true_y, init_w, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R, Dale_law=Dale_law, alpha=10, beta=10, max_iter=max_iter)
        fitW = RNN.build_full_weight_matrix(w, Dale_law=Dale_law)
        unstable, n_unstable = check_unstable(fitW)
        if unstable == False: # stable 
            v = RNN.generate_network_activity(U, T, fitW, b, s, mu0, Q0)
            # print('generated network activity')
        else:
            v = np.zeros((1))
        np.savez(f'models/N={N}_K={K}/EI_control={ei_control}_simulation_{simulation}_J_possibility_{0}', ecll=ecll, ll=ll, initW0=initW0, initW =initW, loss_W = loss_W, w_all = w_all, lossW=lossW, fitW=fitW, b=b, s=s, mu0=mu0, Q0=Q0, C_=C_, d=d, R=R, J=J, true_x=true_x, true_y=true_y, trueA=trueA, true_b=true_b, true_s=true_s, true_mu0=true_mu0, true_Q0=true_Q0, true_C_=true_C_, true_d=true_d, true_R=true_R, v=v)
        
    elif ei_control == 2: # Dale's law and balance but no norm
        Dale_law = 1
        zeta_alpha_beta_gamma_list = [(10**i,0,1,0) for i in list(np.arange(-1,0.5,0.25))]
        initW0, initW, loss_W, w_all = RNN.generate_or_initialize_weights_from_dynamics_LDS(A_target=trueA, R=0.85, zeta_alpha_beta_gamma_list = zeta_alpha_beta_gamma_list, Dale_law=Dale_law)   
        np.savez(f'models/N={N}_K={K}/EI_control={ei_control}_simulation_{simulation}_J_possibility_{0}', initW0=initW0, initW =initW, loss_W = loss_W, w_all = w_all, J=J, trueA=trueA)

    else: # Dale's law and no constraints but also no norm
        Dale_law = 1
        zeta_alpha_beta_gamma_list = [(10**i,0,0,0) for i in list(np.arange(-1,0.5,0.25))]
        initW0, initW, loss_W, w_all = RNN.generate_or_initialize_weights_from_dynamics_LDS(A_target=trueA, R=0.85, zeta_alpha_beta_gamma_list = zeta_alpha_beta_gamma_list, Dale_law=Dale_law)
        np.savez(f'models/N={N}_K={K}/EI_control={ei_control}_simulation_{simulation}_J_possibility_{0}', initW0=initW0, initW =initW, loss_W = loss_W, w_all = w_all, J=J, trueA=trueA)
        
    