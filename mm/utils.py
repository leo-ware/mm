import torch
import numpy as np

def generalized_least_squares(X, y, V):
    cv = torch.inverse(X.T @ torch.inverse(V) @ X)
    return Gaussian(
        mu=cv @ X.T @ torch.inverse(V) @ y,
        sigma=cv
    )

def gaussian_log_pdf(x, mu, sigma):
    k = mu.shape[0]
    return (
        -k/2 * torch.log(2*torch.pi) -
        0.5 * torch.log(torch.det(sigma)) -
        0.5 * (x - mu).T @ torch.inverse(sigma) @ (x - mu)
    )

def is_full_rank(X):
    if not len(X.shape) == 2:
        return False
    return min(X.shape) == torch.linalg.matrix_rank(X)

def is_invertible(X):
    return is_full_rank(X) and (X.shape[0] == X.shape[1])

def listify(x):
    return np.array(x).tolist()

def listify1d(x):
    return np.array(x).reshape(-1).tolist()
