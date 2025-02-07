import numpy as np
import EI_subspace_RNN
import scipy.stats as stats
import scipy.linalg

def generate_eigenvalues(K):
    ''' 
        generate eigenvalue sets in unit disc
    '''
    eigenvalues = []
    i = 0
    # generate random number in square:
    while i < K:
        x, y = np.random.uniform(-1, 1, 2)
        if K % 2 == 1 and i == 0:
            eigenvalues.append(x + 0j)
            i += 1
        elif x**2 + y**2 < 1:
            eigenvalues.append(x + y * 1j)
            eigenvalues.append(x - y * 1j)
            i += 2
    return np.array(eigenvalues)


def generate_dynamics_A(eigenvalues, normal=True, distr='normal'):
    '''
    generate dynamics matrix A with real entries that has a given set of eigenvalues (where complex eigs appear in conjugate pairs)

    eigenvectors: np array
        columns are eigenvectors
    '''
    K = eigenvalues.shape[0]

    # # old way of generating random A (leads to potential large norms and non-normality)
    # comp_A = scipy.linalg.companion(np.poly((eigenvalues))) # companion matrix from characteristic polynomial
    # K = eigenvalues.shape[0]
    # # generate real random matrix for similarity transformation of companion matrix for given eigenvalues
    # P = np.random.rand(K,K) # uniform (0,1)
    # trueA = np.linalg.inv(P)  @ comp_A @ P # similarity

    if K == 1:
        if np.imag(eigenvalues[0])!=0:
            raise Exception('Single eigenvalue should be real')
        else:
            A = np.ones((1,1))
            A[0,0] = np.real(eigenvalues[0])
    else:
        # generating normal A with real entries
        D = np.zeros((K, K)) # real matrix that has eigenvalues of the given set
        i = 0
        while i < K:
            # check if conjugate pairs
            if i == K - 1: # it must be real since it did not have a pair to skip together with
                if np.imag(eigenvalues[i])==0:
                    D[i,i] = np.real(eigenvalues[i])
                    i += 1
                else:
                    raise Exception('Last eigenvalue does not have a pair and is not real')
            elif np.real(eigenvalues[i]) == np.real(eigenvalues[i+1]) and np.imag(eigenvalues[i]) == -np.imag(eigenvalues[i+1]): 
                D[i,i]= np.real(eigenvalues[i])
                D[i+1,i+1]= np.real(eigenvalues[i])
                D[i,i+1]= np.imag(eigenvalues[i])
                D[i+1,i]= -np.imag(eigenvalues[i])
                i += 2
            # check if real when no conjugate pair
            elif np.imag(eigenvalues[i])==0:
                D[i,i] = np.real(eigenvalues[i])
                i += 1
            else:
                raise Exception('Eigenvalues do not have conjugate pairs in right order')
        
        if normal == True:
            S = np.random.normal(0, 1, (K, K))
            Q, R = np.linalg.qr(S)
            Q = Q @ np.diag(np.sign(np.diag(R)))
            A = Q @ D @ Q.T
        else:
            # add values on off diagonal of D to increase non-normality
            num_off_diag = np.random.uniform(0,1) 
            # to get up to maximum number of potential off diagonal terms
            if K % 2 == 0: 
                num_off_diag = num_off_diag * K * (K-2) / 2
            else:
                num_off_diag = num_off_diag * (K-1) * (K-1) / 2
            num_off_diag = int(num_off_diag) + 1
            ind_off_diag = np.random.uniform(0,1, (num_off_diag+2*K, 2)) * K
            ind_off_diag = ind_off_diag.astype(int)
            ind_off_diag.sort(axis=1) # to make sure upper triangular terms
            count_off_diag = 0
            i = 0
            while count_off_diag < num_off_diag and i < num_off_diag+2*K:
                if D[ind_off_diag[i,0],ind_off_diag[i,1]] == 0:
                    if distr == 'normal':
                        D[ind_off_diag[i,0],ind_off_diag[i,1]] = np.random.normal(0, np.sqrt(1/K))
                    elif distr == 'uniform':
                        D[ind_off_diag[i,0],ind_off_diag[i,1]] = np.sin(np.pi * np.random.uniform(-1, 1)) / np.sqrt(1/K)
                    elif distr == 'cauchy':
                        D[ind_off_diag[i,0],ind_off_diag[i,1]] = np.clip(np.random.standard_cauchy(),-2,2) / np.sqrt(1/K)
                    elif distr == 'beta':
                        D[ind_off_diag[i,0],ind_off_diag[i,1]] = (2 * np.random.beta(0.5, 2) - 1) / np.sqrt(1/K)
                    else:
                        raise Exception ('Distribution is not from the accepted group')
                    
                    count_off_diag += 1
                i += 1

            S = np.random.normal(0, 1, (K, K))
            Q, R = np.linalg.qr(S)
            Q = Q @ np.diag(np.sign(np.diag(R)))
            A = Q @ D @ Q.T
    
    # check eigenvalues are matched
    if set(np.round(eigenvalues,5)) != set(np.round(np.linalg.eigvals(A),5)):
        raise Exception ('Eigenvalues of A do not match given set')
    
    # norm_A = np.linalg.norm(A)
    # nonnormality_A = np.linalg.norm(A @ A.T - A.T @ A)
    
    return A

def build_dynamics_matrix_A(W, J):
    return J @ W @ np.linalg.pinv(J)

def mse(z, true_z):
    '''
    mean squared error = 1/datapoints * sum (a-a*)^2
    '''
    n = z.shape[0] * z.shape[1]
    return 1/n * np.trace((z-true_z) @ (z-true_z).T)

def angle_vectors(v1, v2):
    # potentially complex vectors v1 and v2
    cos_angle = np.real(np.vdot(v1,v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    return np.rad2deg(angle_rad)

def norm_complex_scalar(eig):
    eig_norms = np.zeros((eig.shape[0]))
    for i in range(eig_norms.shape[0]):
        eig_norms[i] = np.sqrt(np.real(eig[i])**2 + np.imag(eig[i])**2)
    return eig_norms

def projection_on_vector(v,u):
    ''' 
    projecting v on u
    '''
    return np.dot(u,v)/ (np.linalg.norm(u) ** 2) * u, np.dot(u,v)/ np.linalg.norm(u)

def projection_on_subspace(v,U):
    ''' 
    projecting v on U, U orthogonal
    '''
    v_proj = np.linalg.pinv(U) @ U @ v
    angle = angle_vectors(v, v_proj)
    return v_proj, angle

# TO CHECK!!!
def covariance_alignment(v, J, B):
    ''' 
    B: N x K matrix
    J: K x N matrix
    '''
    # project network on low-dim PC space
    cov_network = v.T @ v # variance of network activity
    proj_v = B.T @ v.T # network activity projected on subspace B
    cov_PCA = proj_v.T @ proj_v # covariance of network in subspace B

    proj_J = J @ v.T
    cov_J = proj_J @ proj_J.T # covariance of network in subspace J

    # covariance of J in subspace B
    proj_J_B = B @ B.T @ J.T
    cov_J_B = proj_J_B @ proj_J_B.T
    
    return np.trace(cov_J) / np.trace(cov_network), np.trace(cov_PCA)/np.trace(cov_network), np.trace(cov_J_B)

def check_unstable(W):
     
    eig = np.linalg.eigvals(W) 
    eig_norms = norm_complex_scalar(eig)

    if len(np.argwhere(eig_norms > 1)) > 0:
        n_unstable = len(np.argwhere(eig_norms > 1))
        return True, n_unstable
    else:
        return False, 0



