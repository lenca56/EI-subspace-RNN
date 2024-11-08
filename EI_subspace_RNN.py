import numpy as np
import scipy.stats as stats
from numpy.random import Generator, PCG64
import utils
from scipy.optimize import minimize, Bounds

class EI_subspace_RNN():
    """
    Class for fitting Excitatory-Inhibitory Recurrent Neural Network with K-dim (low) self-contained dynamics
    like in Lea's paper

    Notation: 
        N_e: number of excitatory units
        N_i: number of inhibitory units
        W_indices: weights indices in (N_e + N_i) x (N_e + N_i) that are non-zero (fixed)
        J: K x (N_e + N_i) projection subspace matrix 
    """

    def __init__(self, N_e, N_i, sparsity, J, seed1, seed2):
        self.N_e, self.N_i, self.sparsity, self.J = N_e, N_i, sparsity, J
        self.K = J.shape[0]
        self.N = self.N_e + self.N_i

        # reproducible random fixed indices for non-zero values of weights
        self.N_weights = int(sparsity * self.N ** 2)
        self.W_ind = [] #np.zeros((self.N_weights,2)).astype(int)
        for i in range(0,self.N):
            self.W_ind.append([i,i])
            #self.W_ind[i,0] = i
            #self.W_ind[i,1] = i

        rng1 = Generator(PCG64(seed1))
        row_rand = rng1.integers(0, self.N, size = 2 * self.N_weights + self.N)
        rng2 = Generator(PCG64(seed2))
        col_rand = rng2.integers(0, self.N, size = 2 * self.N_weights + self.N)

        count = 0
        i = 0
        while count < self.N_weights - self.N:
            if [row_rand[i],col_rand[i]] not in self.W_ind:
                self.W_ind.append([row_rand[i],col_rand[i]])
                count += 1
            i += 1

    def build_full_weight_matrix(self, w):
        W = np.zeros((self.N,self.N))
        if w.shape[0] != self.N_weights:
            raise Exception('Non-zero values of weights do not match in length with non-zero indices of weights')
        for ind in range(self.N_weights):
            if self.W_ind[ind][1] <= self.N_e-1: # excitatory cell
                W[self.W_ind[ind][0], self.W_ind[ind][1]] = w[ind]
            elif self.W_ind[ind][1] >= self.N_e: # inhibitory cell
                W[self.W_ind[ind][0], self.W_ind[ind][1]] = - w[ind]
            else:
                raise Exception('Indices of non-zero values go beyond possible shape')
        return W

    def build_network_covariance(self, s):
        return np.diag(s * np.ones((self.N)))
    
    def build_dynamics_covariance(self, s):
        return self.J @ np.diag(s * np.ones((self.N))) @ self.J.T
    
    def generate_parameters(self, N_weights, D, K):
        ''' 
        Parameters
        ----------
        N_weights: int
            number of non-zero element in connectivity matrix of RNN
        D: int
            dimension of data y_t

        Returns
        -------
        w: N_weights x 1 numpy vector
            non-zero weight values
        b: dict of length 2
            b[0] = K x 1 numpy vector corresponding to input during preparatory period
            b[1] = K x 1 numpy vector corresponding to input during preparatory period
        s: int
            S = np.diag(s) is N x N covariance matrix of Gaussian RNN noise
        mu0: K x 1 numpy vector
            mean of Gaussian distr. of first latent
        Q0: K x K numpy array
            covariance of Gaussiant distr. of first latent
        C_: D x K numpy array
            output mapping from latents x_t to data y_t
        d: D x 1 numpy vector
            offset term for mapping of observations
        R: D x D numpy array
            covariance matrix of Gaussian observation noise
        '''
        w = np.random.normal(1/np.sqrt(self.N), 0.03, N_weights) # TO CHECK WHAT STD TO USE to have scale O(1/sqrt(N))

        s = 0.1

        b1 = np.random.normal(0, 1, K)
        b1 = b1/(b1@b1)
        b1 = b1.reshape((K,1))

        b2 = np.random.normal(0, 1, K)
        b2 = b2/(b2@b2)
        b2 = b2.reshape((K,1))
        b = {0: b1, 1:b2}

        C_ = np.random.normal(2, 1, (D,K))
        d = np.random.normal(3, 1, (D,1))

        mu0 = np.random.normal(0, 0.1, (K,1))
        Q0 = np.random.normal(1, 0.5, (K, K))
        Q0 = np.dot(Q0, Q0.T) # to make P.S.D
        Q0 = 0.5 * (Q0 + Q0.T) # to make symmetric

        R = np.random.normal(1, 0.25, (D, D))
        R = np.dot(R, R.T)
        R = 0.5 * (R + R.T)
        
        return w, b, s, mu0, Q0, C_, d, R
    
    def generate_activity(self, T, w, b, s, mu0, Q0, C_, d, R):
        ''' 
        Parameters
        ----------
        T: number of time points
        '''
        D = C_.shape[0]

        W = self.build_full_weight_matrix(w)
        A = utils.build_dynamics_matrix_A(W, self.J)
        Q = self.build_dynamics_covariance(s)
        t_s = int(T/2)

        x = np.zeros((T, self.K, 1))
        y = np.zeros((T, D, 1))
        x[0] = np.random.multivariate_normal(mu0.flatten(), Q0).reshape((self.K,1))
        y[0] = C_ @ x[0] + d
        for i in range(1, T):
            x[i] = np.random.multivariate_normal( (A @ x[i-1] + b[i-1 >= t_s]).reshape((self.K)), Q).reshape((self.K,1))
            y[i] = np.random.multivariate_normal((C_ @ x[i] + d).reshape(D), R).reshape((D,1))
            
        return x, y
    
    def Kalman_filter_E_step(self, y, w, b, s, mu0, Q0, C_, d, R):

        W = self.build_full_weight_matrix(w)
        A = utils.build_dynamics_matrix_A(W, self.J)
        Q = self.build_dynamics_covariance(s)
        T = y.shape[0]
        t_s = int(T/2) # assume switch from preparatory to movement happens midway
        
        mu = np.zeros((T, self.K, 1))
        mu_prior = np.zeros((T, self.K, 1))
        V = np.zeros((T, self.K, self.K))
        V_prior = np.zeros((T, self.K, self.K))
        
        # first step
        mu_prior[0] = mu0
        V_prior[0] = Q0
        V[0] = np.linalg.inv(C_.T @ np.linalg.inv(R) @ C_  + np.linalg.inv(V_prior[0]))
        mu[0] = V[0] @ (C_.T @ np.linalg.inv(R) @ (y[0] - d) + np.linalg.inv(V_prior[0]) @ mu_prior[0])
        
        for t in range (1,T):
            # prior update
            mu_prior[t] = A @ mu_prior[t-1] + b[t-1 >= t_s]
            V_prior[t] = A @ V[t-1] @ A.T + Q

            # filter update
            V[t] = np.linalg.inv(C_.T @ np.linalg.inv(R) @ C_  + np.linalg.inv(V_prior[t]))
            mu[t] = V[t] @ (C_.T @ np.linalg.inv(R) @ (y[t] - d) + np.linalg.inv(V_prior[t]) @ mu_prior[t])

        return mu, mu_prior, V, V_prior

    def Kalman_smoother_E_step(self, A, mu, mu_prior, V, V_prior):
        T = mu.shape[0]
    
        m = np.zeros((T, self.K, 1))
        cov = np.zeros((T, self.K, self.K))
        cov_next = np.zeros((T-1, self.K, self.K))

        # last step (equal to last Kalman filter output)
        m[-1] = mu[-1]
        cov[-1] = V[-1]

        for t in range (T-2,-1,-1):
            # auxilary matrix
            L = V[t] @ A.T @ np.linalg.inv(V_prior[t+1])

            # smoothing updates
            m[t] = mu[t] + L @ (m[t+1] - mu_prior[t+1])
            cov[t] = V[t] + L @ (cov[t+1] - V_prior[t+1]) @ L.T
            cov_next[t] = L @ cov[t+1]

        return m, cov, cov_next

    def closed_form_M_step(self, y, w, m, cov, cov_next):
        ''' 
        closed-form updates for all parameters except the weights
        '''
        W = self.build_full_weight_matrix(w)
        A = utils.build_dynamics_matrix_A(W, self.J)

        T = y.shape[0]
        t_s = int(T/2)
        M1 = np.sum(m, axis=0)
        M1_T = np.sum(cov, axis=0)
        M_next = np.sum(cov_next, axis=0)
        Y1 = np.sum(y, axis=0)
        Y2 = np.zeros((y.shape[1], y.shape[1]))
        Y_tilda = np.zeros((y.shape[1], self.K))
        for t in range(0,T):
            M1_T = M1_T + m[t] @ m[t].T
            Y_tilda = Y_tilda + y[t] @ m[t].T
            Y2 = Y2 + y[t] @ y[t].T
            if t != T-1:
                M_next = M_next + m[t] @ m[t+1].T
        
        # updates first latent
        mu0 = m[0]
        Q0 = cov[0]

        # updates observation parameters
        C_ = (Y1 @ M1.T - T * Y_tilda) @ np.linalg.inv(M1 @ M1.T - T * M1_T)
        d = 1/T * (Y1 - C_ @ M1)
        R = 1/T * (Y2 + T * d @ d.T - d @ Y1.T - Y1 @ d.T - Y_tilda @ C_.T - C_ @ Y_tilda.T + d @ M1.T @ C_.T + C_ @ M1 @ d.T + C_ @ M1_T @ C_.T)

        # updates dynamics parameters
        b = {0:'', 1:''}
        b[0] = 1/t_s * (np.sum(m[1:t_s+1], axis=0) - A @ np.sum(m[0:t_s], axis=0))
        b[1] = 1/(T - t_s - 1) * (np.sum(m[t_s+1:T], axis=0) - A @ np.sum(m[t_s:T-1], axis=0))
        J_aux = np.linalg.inv(self.J @ self.J.T)
        s = np.trace(J_aux @ (M1_T - cov[0] - m[0] @ m[0].T + A @ (M1_T - cov[-1] - m[-1] @ m[-1].T) @ A.T - 2 * A @ M_next))
        s_aux = np.zeros((self.K, self.K))
        for t in range(0,T-1):
            s_aux = s_aux + b[t >= t_s] @ b[t >= t_s].T - 2 * b[t >= t_s] @ (m[t+1].T - m[t].T @ A.T)
        s = s + np.trace(J_aux @ s_aux)
        s = s / (self.K * (T-1))

        return b, s, mu0, Q0, C_, d, R
    
    def loss_weights_M_step(self, w, s, b, m, cov, cov_next, alpha=1, beta=1):
        '''
        gradient of loss function of weights to be minimized
        '''
        W = self.build_full_weight_matrix(w)
        A = utils.build_dynamics_matrix_A(W, self.J)

        T = m.shape[0]
        t_s = int(T/2)
        aux = np.zeros((self.K, self.K))
        M1_T1 = np.sum(cov[:-1], axis=0)
        M_next = np.sum(cov_next, axis=0)
        for t in range(0,T-1):
            M1_T1 = M1_T1 + m[t] @ m[t].T
            M_next = M_next + m[t] @ m[t+1].T
            aux = aux + b[t >= t_s] @ m[t].T

        J_aux = np.linalg.inv(self.J @ self.J.T)
        loss_W = - 1/s * - np.trace(A.T @ J_aux @ aux) 
        loss_W = loss_W - 1/s * - 0.5 * np.trace(A.T @ J_aux @ A @ M1_T1)
        loss_W = loss_W - 1/s * np.trace(J_aux @ A @ M_next)

        # regularization
        Jpinv_aux = self. J @ W @ (np.identity((self.N)) - np.linalg.pinv(self.J) @ self.J)
        loss_W = loss_W + 0.5 * alpha * np.trace(Jpinv_aux @ Jpinv_aux.T)
        loss_W = loss_W + 0.5 * beta * np.ones((self.N,1)).T @ W.T @ W @ np.ones((self.N,1)) 
        return loss_W
    
    def gradient_weights_M_step(self, w, s, b, m, cov, cov_next, alpha=1, beta=1):
        '''
        gradient of loss function of weights to be minimized
        '''
        W = self.build_full_weight_matrix(w)
        A = utils.build_dynamics_matrix_A(W, self.J)

        T = m.shape[0]
        t_s = int(T/2)
        aux = np.zeros((self.K, self.K))
        M1_T1 = np.sum(cov[:-1], axis=0)
        M_next = np.sum(cov_next, axis=0)
        for t in range(0,T-1):
            M1_T1 = M1_T1 + m[t] @ m[t].T
            M_next = M_next + m[t] @ m[t+1].T
            aux = aux + b[t >= t_s] @ m[t].T

        grad_W = 1/s * self.J.T @ np.linalg.inv(self.J @ self.J.T) @ (- aux - A @ M1_T1 + M_next.T) @ np.linalg.pinv(self.J).T
        grad_W = grad_W - alpha * self.J.T @ self.J @ W @ (np.identity((self.N)) - np.linalg.pinv(self.J) @ self.J)
        grad_W = grad_W - beta * W @ np.ones((self.N,1)) @ np.ones((self.N,1)).T
        return - grad_W
    
    def fit_EM(self, x, y, init_w, init_b, init_s, init_mu0, init_Q0, init_C_, init_d, init_R, alpha=1, beta=1, max_iter=300):
        
        w = np.copy(init_w)
        b = init_b.copy()
        s = np.copy(init_s)
        mu0 = np.copy(init_mu0)
        Q0 = np.copy(init_Q0)
        C_ = np.copy(init_C_)
        d = np.copy(init_d)
        R = np.copy(init_R)

        for iter in range(max_iter):
            W = self.build_full_weight_matrix(w)
            A = utils.build_dynamics_matrix_A(W, self.J)

            # E-step
            mu, mu_prior, V, V_prior = self.Kalman_filter_E_step(y, w, b, s, mu0, Q0, C_, d, R)
            m, cov, cov_next = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)

            # M-step
            b, s, mu0, Q0, C_, d, R = self.closed_form_M_step(y, w, m, cov, cov_next)
            opt_fun = lambda w: self.loss_weights_M_step(w, s, b, m, cov, cov_next, alpha, beta)
            # opt_grad = lambda w: self.gradient_weights_M_step(w, s, b, m, cov, cov_next, alpha, beta)
            bounds = [(0, None) for i in range(w.shape[0])]
            def constraint(w):
                return w.flatten()
            con = {'type': 'ineq', 'fun': constraint}
            w = minimize(opt_fun, w.flatten(), method='SLSQP', constraints=[con], bounds=bounds).x
            w = w.reshape((w.shape[0], 1))
        
        return w, b, s, mu0, Q0, C_, d, R




        
    
    