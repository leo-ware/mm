import torch

class Gaussian:
    def __init__(self, mean, covariance):
        if isinstance(mean, Gaussian):
            covariance = mean.covariance + covariance
            mean = mean.mean  
        self.mean = mean.reshape(-1, 1).to(float)
        self.covariance = covariance.to(float)
        self.k = mean.shape[0]
    
    def pdf(self, x):
        x = x.reshape(-1, 1)
        return (
            (2*torch.pi)**(-self.k/2) *
            torch.det(self.covariance)**(-0.5) *
            torch.exp(-0.5 * ((x - self.mean).T @ torch.inverse(self.covariance) @ (x - self.mean)))
        )
    
    def log_pdf(self, x):
        x = x.reshape(-1, 1)
        return (
            -self.k/2 * torch.log(torch.tensor(2*torch.pi)) -
            0.5 * torch.log(torch.det(self.covariance)) -
            0.5 * (x - self.mean).T @ torch.inverse(self.covariance) @ (x - self.mean)
        )
    
    def log_likelihood(self, x):
        x = x.reshape(-1, 1)
        return - 0.5 * (x - self.mean).T @ torch.inverse(self.covariance) @ (x - self.mean)
    
    def __add__(self, other):
        if isinstance(other, Gaussian):
            return Gaussian(
                self.mean + other.mean,
                self.covariance + other.covariance
            )
        else:
            return Gaussian(self.mean + other, self.covariance)
    
    def __radd__(self, other):
        return self + other
    
    def __rmatmul__(self, A):
        A = A.to(float)
        return Gaussian(
            A @ self.mean,
            A @ self.covariance @ A.T
        )
