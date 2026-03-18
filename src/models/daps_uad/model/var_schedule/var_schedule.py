import torch

class EDMSchedule:

    def __init__(
    self,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    sigma_data: float = 0.5,
    rho: float = 7,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def time_steps(self,step):
        return (
                self.sigma_max ** (1 / self.rho)
                + step
                * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            ) ** self.rho

    def sigma(self, step):
        return self.time_steps(step)

    def mean_factor(self,sigma):
        return 1/torch.sqrt(sigma**2 + self.sigma_data**2)
    
    def loss_weight(self,sigma):
        return (self.sigma_data**2 + sigma**2)/(sigma * self.sigma_data)**2

    def sample_sigma(self,x):
        return torch.exp(self.P_mean + self.P_std * torch.randn((x.shape[0],1,1,1)))

    def c_skip(self, sigma):
        return self.sigma_data**2/(sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return (sigma * self.sigma_data)/torch.sqrt(self.sigma_data**2 + sigma**2)
    
    def c_in(self, sigma):
        return self.mean_factor(sigma)
    
    def c_noise(self, sigma):
        return 0.25 * torch.log(sigma)
