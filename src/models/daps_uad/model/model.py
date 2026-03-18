from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict

class SDE_edm(nn.Module):

    def __init__(self,
                network : nn.Module,
                schedule,
                model_config):
        super().__init__()

        self._init_config(model_config)
        self.network = network
        self.schedule = schedule


    def _init_config(self, model_config: Dict):
        pass

    def train_step(self, dict_ord: Dict):

        x = dict_ord["image"]

        sigma = self.schedule.sample_sigma(x).to(x.device)
        
        c_skip = self.schedule.c_skip(sigma)
        c_out = self.schedule.c_out(sigma)
        c_in = self.schedule.c_in(sigma)
        c_noise = self.schedule.c_noise(sigma)

        noise = torch.randn_like(x)

        x_noise = x + sigma * noise

        f_pred = self.network(c_in * x_noise,c_noise.view(-1,))

        f_target = (x - c_skip * x_noise) / c_out

        loss_weight = self.schedule.loss_weight(sigma)

        mse = (f_pred - f_target).pow(2).mean(dim=(1,2,3))

        loss = (loss_weight * mse).mean()
        if not torch.isfinite(loss):
            print(f"[LOSS] Non-finite loss encountered: {loss.item():.3e}")
            return torch.tensor(0.0, device=x.device)

        return loss
    
    @torch.no_grad()
    def drift(self, x: torch.Tensor, sigma: torch.Tensor):
        """
        EDM probability-flow ODE drift:
            dx/dsigma = -(d(x,sigma) - x) / sigma
        """

        # Get EDM coefficients
        c_skip = self.schedule.c_skip(sigma)
        c_out  = self.schedule.c_out(sigma)
        c_in   = self.schedule.c_in(sigma)
        c_noise = self.schedule.c_noise(sigma)

        # Predict residual f
        f_pred = self.network(c_in * x, c_noise.view(-1,))

        # EDM denoiser
        d_pred = c_skip * x + c_out * f_pred

        # Probability-flow ODE drift
        drift = (x - d_pred) / (sigma + 1e-6)

        return drift

    @torch.no_grad()
    def predict_x0(self, x_sigma: torch.Tensor, sigma: torch.Tensor):
        """
        EDM denoiser:
            x0_hat = c_skip * x_sigma + c_out * f(x_sigma)
        """
        c_skip = self.schedule.c_skip(sigma)
        c_out  = self.schedule.c_out(sigma)
        c_in   = self.schedule.c_in(sigma)
        c_noise = self.schedule.c_noise(sigma)

        f_pred = self.network(c_in * x_sigma, c_noise.view(-1,))
        x0_hat = c_skip * x_sigma + c_out * f_pred
        return x0_hat
    
    @torch.no_grad()
    def score(self, x0: torch.Tensor, sigma: torch.Tensor):
        """
        Score wrt x0 under EDM forward process:
            score(x0; t) = (predict_x_sigma(x0) - x0) / sigma^2
        """
        # Convert x0 to a noisy x_t that would produce x0 under EDM
        # Forward process:  x_t = x0 + sigma * noise
        # But for score, the conditional mean is:
        #     mean[x_t | x0] ≈ x0
        # So EDM uses the denoiser identity:

        # Re-encode x0 back to x_t mean:
        x_t_mean = x0  # since forward noise is zero-mean

        # Decode it:
        x0_hat = self.predict_x0(x_t_mean, sigma)

        score = (x0_hat - x0) / (sigma ** 2 + 1e-8)
        return score
    
    @torch.no_grad()
    def sample_ode_from_xt(self, x_t: torch.Tensor, sigma: torch.Tensor, steps: int =4):
        """
        Deterministic EDM ODE sampler applied locally around (x_t, sigma).

        Inputs:
            x_t   : noisy input at noise level sigma (B,C,H,W)
            sigma : (B,1,1,1) noise level corresponding to t
            steps : number of small ODE refinement steps

        Output:
            x0 : deterministic reconstruction through local ODE integration
        """

        sigma_scalar = sigma.view(-1)[0].item()   # extract value
        sigma_min = self.schedule.sigma_min

        # For small steps, linearly interpolate σ
        sigma_steps = torch.linspace(
            sigma_scalar, sigma_min, steps, device=x_t.device
        )  # shape: (steps,)

        x = x_t.clone()

        for i in range(steps - 1):
            # shape (1,1,1,1), will broadcast over (B,C,H,W)
            sigma_i = sigma_steps[i].view(1, 1, 1, 1)
            sigma_next = sigma_steps[i + 1].view(1, 1, 1, 1)
            ds = sigma_next - sigma_i

            drift = self.drift(x, sigma_i)   # dx/dσ, sigma_i broadcasts
            x = x + drift * ds               # Euler step, ds broadcasts

        return x




    