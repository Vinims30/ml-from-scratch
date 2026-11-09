import numpy as np

def euclidean_two_loops(A, B):
    D = np.zeros((A.shape[0], B.shape[0]))
    for i, linha in enumerate(A):
        for j, linhaj in enumerate(B):
            D[i,j] = np.sqrt(np.sum((linha - linhaj) **2))
    return D

def euclidean_one_loop(A, B):
    D = np.zeros((A.shape[0], B.shape[0]))
    for i, linha in enumerate(A):
        D[i] = np.sqrt(np.sum((linha - B) ** 2, axis=1))
    return D

def euclidean_vectorized(A, B):
    D2 =  np.sum(A ** 2, axis=1).reshape(-1,1) + (np.sum(B ** 2,axis=1)  - 2 *(A @ B.T))
    D2 = np.maximum(D2,0)
    D = np.sqrt(D2)
    return D