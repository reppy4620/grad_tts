import torch

class SDE:
    def __init__(self, beta_min=0.05, beta_max=20):
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def sde(self, x_t, mu, t):
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = 0.5 * (mu - x_t) * beta_t[:, None, None, None]
        diffusion = torch.sqrt(beta_t)[:, None, None, None]
        return drift, diffusion
    
    def marginal_prob(self, x_0, mu, t):
        beta_int = self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t ** 2
        c = torch.exp(-0.5 * beta_int)[:, None, None, None]
        mean = c * x_0 + (1 - c) * mu
        std = torch.sqrt(1. - torch.exp(-beta_int))[:, None, None, None]
        return mean, std

    def reverse_sde(self, score, x_t, mu, t):
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = (0.5 * (mu - x_t) - score) * beta_t[:, None, None, None]
        diffusion = beta_t[:, None, None, None]
        return drift, diffusion

    def probability_flow(self, score, x_t, mu, t):
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = 0.5 * (mu - x_t - score) * beta_t[:, None, None, None]
        diffusion = torch.zeros_like(diffusion)
        return drift, diffusion
