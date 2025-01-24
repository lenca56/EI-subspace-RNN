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
        x, y = np.random.uniform(0, 1, 2)
        if K % 2 == 1 and i == 0:
            eigenvalues.append(x + 0j)
            i += 1
        elif x**2 + y**2 < 1:
            eigenvalues.append(x + y * 1j)
            eigenvalues.append(x - y * 1j)
            i += 2
    return np.array(eigenvalues)


def generate_dynamics_A(eigenvalues):
    '''
    generate dynamics matrix A with real entries that has a given set of eigenvalues (where complex eigs appear in conjugate pairs)

    eigenvectors: np array
        columns are eigenvectors
    '''
    n_eig = eigenvalues.shape[0]

    # # old way of generating random A (leads to potential large norms and non-normality)
    # comp_A = scipy.linalg.companion(np.poly((eigenvalues))) # companion matrix from characteristic polynomial
    # K = eigenvalues.shape[0]
    # # generate real random matrix for similarity transformation of companion matrix for given eigenvalues
    # P = np.random.rand(K,K) # uniform (0,1)
    # trueA = np.linalg.inv(P)  @ comp_A @ P # similarity

    if n_eig == 1:
        if np.imag(eigenvalues[0])!=0:
            raise Exception('Single eigenvalue should be real')
        else:
            A = np.ones((1,1))
            A[0,0] = np.real(eigenvalues[0])
    else:
        # generating normal A with real entries
        D = np.zeros((n_eig, n_eig)) # real matrix that has eigenvalues of the given set
        i = 0
        while i < n_eig:
            # check if conjugate pairs
            if np.real(eigenvalues[i]) == np.real(eigenvalues[i+1]) and np.imag(eigenvalues[i]) == -np.imag(eigenvalues[i+1]): 
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
        
        S = np.random.normal(0, 1, (n_eig, n_eig))
        Q, R = np.linalg.qr(S)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        A = Q @ D @ Q.T
    return A

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
    # project network on low-dim PC space
    cov_network = v.T @ v
    proj_v = v @ B
    cov_PCA = proj_v.T @ proj_v

    proj_J = J @ v.T
    cov_J = proj_J @ proj_J.T
    
    return np.trace(cov_J) / np.trace(cov_network), np.trace(cov_PCA)/np.trace(cov_network)


