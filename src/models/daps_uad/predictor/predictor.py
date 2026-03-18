import torch
import logging
import torch.nn as nn
import math
log = logging.getLogger(__name__)

class Predictor:
    def __init__(
        self,
        model: nn.Module,
        inference_scheduler: nn.Module,
        predictor_config,
    ):
        self._init_config(predictor_config)

        self.model = model
        self.model.eval()

        self.inference_scheduler = inference_scheduler

        self.to(self.device)

    def _init_config(self, predictor_config):
        self.N = predictor_config.N
        self.N_inner = predictor_config.N_inner

        self.lam_logit_init = predictor_config.lam_logit_init 
        self.lam_smooth = predictor_config.lam_smooth

        self.alpha = predictor_config.alpha

        self.ode_steps = predictor_config.ode_steps
        self.c_anneal = predictor_config.c_anneal

        dev = predictor_config.device
        self.device = torch.device(dev) if not isinstance(dev, torch.device) else dev

        self.mu_logit = predictor_config.mu_logit

        self.use_auto_lam_logit = getattr(predictor_config, "use_auto_lam_logit", True)
        self.delta_a = getattr(predictor_config, "delta_a", 1e-3)
        self.delta_x = getattr(predictor_config, "delta_x", 5e-3)

        self.eta_a_min = getattr(predictor_config, "eta_a_min", 1e-6)
        self.eta_a_max = getattr(predictor_config, "eta_a_max", 1e-2)

    def to(self, device):
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self.model.to(self.device)
        if hasattr(self.inference_scheduler, "to"):
            self.inference_scheduler.to(self.device)
        return self

    @torch.no_grad()
    def sigmoid_clamped(self, a, eps=1e-6):
        return torch.sigmoid(a).clamp(eps, 1 - eps)

    @torch.no_grad()
    def safe_sigmoid(self, a, eps=1e-6):
        return self.sigmoid_clamped(a, eps=eps)

    @torch.no_grad()
    def grad_rms(self, x, eps=1e-8):
        return torch.sqrt(torch.mean(x * x) + eps)

    @torch.no_grad()
    def grad_x0_energy(self, x0, x0_hat, a, y_k, sigma_prior, lambda_obs, eps=1e-8):
        """
        All inputs are batched tensors with the same shape as y:
        (B, C, H, W) after broadcasting.
        """
        M = self.sigmoid_clamped(a)
        lam2 = (lambda_obs * lambda_obs).clamp_min(1e-8)
        sig2 = (sigma_prior * sigma_prior).clamp_min(1e-8)

        inv_sig2 = 1.0 / sig2
        inv_lam2 = 1.0 / lam2
        grad_prior = (x0 - x0_hat) * inv_sig2
        grad_meas = (M * M) * (x0 - y_k) * inv_lam2
        return grad_prior + grad_meas

    @torch.no_grad()
    def grad_a_parts(
        self,
        x0,
        a,
        y_k,
        lambda_obs,
        lam_logit,
        lam_smooth,
        mu_logit,
        eps=1e-8,
    ):
        """
        All tensors are batched: (B, C, H, W) except scalars (lam_logit, lam_smooth, mu_logit).
        Broadcasting handles batch.
        Returns:
            grad_total, grad_meas, grad_logit, grad_smooth, grad_log
        """
        M = self.sigmoid_clamped(a)
        r = y_k - x0
        lam2 = (lambda_obs * lambda_obs).clamp_min(1e-8)

        inv_lam2 = 1.0 / lam2

        # measurement part
        grad_meas = M * M * (1.0 - M) * (r * r) * inv_lam2

        # barrier -sum log M (gradient wrt a)
        grad_log = -(1.0 - M)

        # quadratic prior in logits
        grad_logit = lam_logit * (a - mu_logit)

        grad_smooth = torch.zeros_like(a)
        if lam_smooth != 0.0:
            grad_smooth[:, :, :-1, :] += lam_smooth * (a[:, :, :-1, :] - a[:, :, 1:, :])
            grad_smooth[:, :, 1:, :] += lam_smooth * (a[:, :, 1:, :] - a[:, :, :-1, :])
            grad_smooth[:, :, :, :-1] += lam_smooth * (a[:, :, :, :-1] - a[:, :, :, 1:])
            grad_smooth[:, :, :, 1:] += lam_smooth * (a[:, :, :, 1:] - a[:, :, :, :-1])

        grad_total = grad_meas + grad_logit + grad_smooth + grad_log
        return grad_total, grad_meas, grad_logit, grad_smooth, grad_log

    @torch.no_grad()
    def langevin_update(
        self, x, grad, eta, add_noise=True, clip_grad=None, clip_x=None
    ):
        """
        x, grad: (B, C, H, W)
        eta: scalar or tensor broadcastable to x
        """
        if clip_grad is not None:
            grad = torch.clamp(grad, -clip_grad, clip_grad)

        eta = torch.as_tensor(eta, device=x.device, dtype=x.dtype)
        noise = torch.randn_like(x) if add_noise else 0.0
        x_new = x - eta * grad + torch.sqrt(2.0 * eta) * noise

        if clip_x is not None:
            x_new = torch.clamp(x_new, -clip_x, clip_x)
        return x_new

    @torch.no_grad()
    def rms_normalized_stepsize(self, grad, delta, eta_min, eta_max, eps=1e-8):
        """
        grad: (B, C, H, W)
        Returns a scalar eta shared across the batch
        """
        eta = delta / (self.grad_rms(grad, eps=eps) + eps)
        return torch.clamp(eta, eta_min, eta_max)

    @torch.no_grad()
    def auto_lam_logit_grad_balance(
        self,
        a,
        g_meas,  
        mu_logit,
        alpha=1.0,  
        lam_min=1e-4,
        lam_max=1e2,
        lam_prev=None,
        ema_beta=0.9,
        eps=1e-8,
    ): 
        b = a - mu_logit
        rms_meas = self.grad_rms(g_meas, eps=eps)
        rms_b = self.grad_rms(b, eps=eps)

        lam_raw = rms_meas / (alpha * (rms_b + eps) + eps)
        lam_raw = torch.clamp(lam_raw, lam_min, lam_max)

        if lam_prev is None:
            lam = lam_raw
        else:
            lam = ema_beta * lam_prev + (1.0 - ema_beta) * lam_raw
            lam = torch.clamp(lam, lam_min, lam_max)

        return lam, lam_raw, rms_meas, rms_b


    @torch.no_grad()
    def predict(self, dict_ord):
        """
        dict_ord["image"]: (B, C, H, W) — batch of images.
        Everything is kept fully batched throughout.
        """
        y = dict_ord["image"].to(self.device)

        device = y.device
        dtype = y.dtype
        B = y.shape[0]

        t_grid = torch.linspace(0.0, 0.99, steps=self.N, device=device)

        a = torch.full_like(y, float(self.mu_logit))

        # t=0
        t0_tensor = torch.full(
            (B, 1, 1, 1), float(t_grid[0]), device=device, dtype=dtype
        )
        sigma0 = self.inference_scheduler.time_steps(
            t0_tensor
        )  

        while sigma0.dim() < y.dim():
            sigma0 = sigma0.unsqueeze(-1)

        x_t = sigma0 * torch.randn_like(y) + y

        lam_logit = torch.as_tensor(self.lam_logit_init, device=device, dtype=dtype)
        lam_logit_ema = lam_logit.clone()
        lambda_floor = 5e-3

        for k, t_scalar in enumerate(t_grid):
            t_val = float(t_scalar)
            t_tensor = torch.full((B, 1, 1, 1), t_val, device=device, dtype=dtype)
            sigma = self.inference_scheduler.time_steps(t_tensor)

            # broadcast sigma to (B, C, H, W) if needed
            while sigma.dim() < y.dim():
                sigma = sigma.unsqueeze(-1)

            # prior mean from diffusion, fully batched
            x0_hat = self.model.sample_ode_from_xt(x_t, sigma, steps=self.ode_steps)

            x0 = 0.5 * (y.clone() + x0_hat.clone())

            lambda_k =  torch.clamp(self.c_anneal * (sigma), min=lambda_floor)
            r_k = lambda_k
            
            grad_total, _, _, _, g_log = self.grad_a_parts(
                x0=x0,
                a=a,
                y_k=y,
                lambda_obs=torch.clamp(lambda_k, 5e-2), # clamp for numerical stability
                lam_logit=torch.as_tensor(0.0, device=device, dtype=dtype),
                lam_smooth=0.0,
                mu_logit=self.mu_logit,
            )

            g_meas_base = grad_total - g_log

            if self.use_auto_lam_logit:
                lam_logit_ema, lam_logit_raw, rms_meas, rms_b = (
                    self.auto_lam_logit_grad_balance(
                        a=a,
                        g_meas=g_meas_base,
                        mu_logit=self.mu_logit,
                        alpha=self.alpha,
                        lam_min=1e-3,
                        lam_max=1e2,
                        lam_prev=lam_logit_ema,
                        ema_beta=0.95,
                    )
                )
                lam_logit = lam_logit_ema

            for inner in range(self.N_inner):
                # 1) measurement-only grad wrt a (for auto lam tuning)
                

                # 2) full gradient with current lam_logit and lam_smooth
                ga, g_meas, g_logit, g_smooth, g_log = self.grad_a_parts(
                    x0=x0,
                    a=a,
                    y_k=y,
                    lambda_obs=torch.clamp(lambda_k, 5e-2),
                    lam_logit=lam_logit,
                    lam_smooth=self.lam_smooth,
                    mu_logit=self.mu_logit,
                )

            
                eta_a = self.rms_normalized_stepsize(
                    ga, delta= self.delta_a, eta_min=self.eta_a_min, eta_max=self.eta_a_max
                )

                a = self.langevin_update(
                    a,
                    ga,
                    eta=eta_a,
                    add_noise=True,
                    clip_grad=50,
                    clip_x=None,
                )

                # --- x0 full step ---
                gx0 = self.grad_x0_energy(x0, x0_hat, a, y, r_k, lambda_k)

                eta_x = self.rms_normalized_stepsize(
                    gx0, delta=self.delta_x, eta_min=1e-5, eta_max=2e-2
                )

                x0 = self.langevin_update(
                    x0,
                    gx0,
                    eta=eta_x,
                    add_noise=True if k < self.N - 1 else False,
                    clip_grad=8e3,
                    clip_x=None,
                )

            if k < len(t_grid) - 1:
                t_next = float(t_grid[k + 1])
                t_next_tensor = torch.full(
                    (B, 1, 1, 1), t_next, device=device, dtype=dtype
                )
                sigma_next = self.inference_scheduler.time_steps(t_next_tensor)
                while sigma_next.dim() < y.dim():
                    sigma_next = sigma_next.unsqueeze(-1)
                x_t = x0 + sigma_next * torch.randn_like(x0)

        dict_pred = {
            "reconstruction": x0,  # (B, C, H, W)
            "anomaly_map": 1 - self.safe_sigmoid(a),
            "anomaly_map_2": (x0 - y).square()
        }

        log.debug(f"[Predictor] Prediction complete for batch of size {y.shape[0]}")
        return dict_pred


class PredictorAblationA:
    # Ablation No mask
    def __init__(
        self,
        model: nn.Module,
        inference_scheduler: nn.Module,
        predictor_config,
    ):
        self._init_config(predictor_config)

        self.model = model
        self.model.eval()

        self.inference_scheduler = inference_scheduler

        self.to(self.device)

    def _init_config(self, predictor_config):
        self.N = predictor_config.N
        self.N_inner = predictor_config.N_inner

        self.lam_logit_init = predictor_config.lam_logit_init
        self.lam_smooth = predictor_config.lam_smooth
        self.alpha = predictor_config.alpha

        self.ode_steps = predictor_config.ode_steps
        self.c_anneal = predictor_config.c_anneal

        # device can be "cuda", "cuda:0", "cpu", or a torch.device
        dev = predictor_config.device
        self.device = torch.device(dev) if not isinstance(dev, torch.device) else dev

        self.mu_logit = predictor_config.mu_logit

        # these are referenced in predict(), so make sure they exist
        self.use_auto_lam_logit = getattr(predictor_config, "use_auto_lam_logit", True)
        self.delta_a = getattr(predictor_config, "delta_a", 1e-3)
        self.delta_x = getattr(predictor_config, "delta_x", 1e-3)

        self.eta_a_min = getattr(predictor_config, "eta_a_min", 1e-6)
        self.eta_a_max = getattr(predictor_config, "eta_a_max", 1e-2)

    def to(self, device):
        """Move model & scheduler to device and update self.device."""
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self.model.to(self.device)
        if hasattr(self.inference_scheduler, "to"):
            self.inference_scheduler.to(self.device)
        return self

    @torch.no_grad()
    def sigmoid_clamped(self, a, eps=1e-6):
        return torch.sigmoid(a).clamp(eps, 1 - eps)

    # optional alias to keep your original dict_pred line working
    @torch.no_grad()
    def safe_sigmoid(self, a, eps=1e-6):
        return self.sigmoid_clamped(a, eps=eps)

    @torch.no_grad()
    def grad_rms(self, x, eps=1e-8):
        # RMS over all elements (B, C, H, W)
        return torch.sqrt(torch.mean(x * x) + eps)

    @torch.no_grad()
    def grad_x0_energy(self, x0, x0_hat, a, y_k, sigma_prior, lambda_obs, eps=1e-8):
        """
        All inputs are batched tensors with the same shape as y:
        (B, C, H, W) after broadcasting.
        """
        M = self.sigmoid_clamped(a)
        inv_sig2 = 1.0 / (sigma_prior * sigma_prior + eps)
        inv_lam2 = 1.0 / (lambda_obs * lambda_obs + eps)
        grad_prior = (x0 - x0_hat) * inv_sig2
        grad_meas = (M * M) * (x0 - y_k) * inv_lam2
        return grad_prior + grad_meas

    @torch.no_grad()
    def grad_a_parts(
        self,
        x0,
        a,
        y_k,
        lambda_obs,
        lam_logit,
        lam_smooth,
        mu_logit,
        eps=1e-8,
    ):
        """
        All tensors are batched: (B, C, H, W) except scalars (lam_logit, lam_smooth, mu_logit).
        Broadcasting handles batch.
        Returns:
            grad_total, grad_meas, grad_logit, grad_smooth, grad_log
        """
        M = self.sigmoid_clamped(a)
        r = y_k - x0
        inv_lam2 = 1.0 / (lambda_obs * lambda_obs + eps)

        # measurement part
        grad_meas = M * M * (1.0 - M) * (r * r) * inv_lam2

        # barrier -sum log M (gradient wrt a)
        grad_log = -(1.0 - M)

        # quadratic prior in logits
        grad_logit = lam_logit * (a - mu_logit)

        # anisotropic smoothness
        grad_smooth = torch.zeros_like(a)
        if lam_smooth != 0.0:
            grad_smooth[:, :, :-1, :] += lam_smooth * (a[:, :, :-1, :] - a[:, :, 1:, :])
            grad_smooth[:, :, 1:, :] += lam_smooth * (a[:, :, 1:, :] - a[:, :, :-1, :])
            grad_smooth[:, :, :, :-1] += lam_smooth * (a[:, :, :, :-1] - a[:, :, :, 1:])
            grad_smooth[:, :, :, 1:] += lam_smooth * (a[:, :, :, 1:] - a[:, :, :, :-1])

        grad_total = grad_meas + grad_logit + grad_smooth + grad_log
        return grad_total, grad_meas, grad_logit, grad_smooth, grad_log

    @torch.no_grad()
    def langevin_update(
        self, x, grad, eta, add_noise=True, clip_grad=None, clip_x=None
    ):
        """
        x, grad: (B, C, H, W)
        eta: scalar or tensor broadcastable to x
        """
        if clip_grad is not None:
            grad = torch.clamp(grad, -clip_grad, clip_grad)

        eta = torch.as_tensor(eta, device=x.device, dtype=x.dtype)
        noise = torch.randn_like(x) if add_noise else 0.0
        x_new = x - eta * grad + torch.sqrt(2.0 * eta) * noise

        if clip_x is not None:
            x_new = torch.clamp(x_new, -clip_x, clip_x)
        return x_new

    @torch.no_grad()
    def rms_normalized_stepsize(self, grad, delta, eta_min, eta_max, eps=1e-8):
        """
        grad: (B, C, H, W)
        Returns a scalar eta shared across the batch
        (but still fully batch-compatible).
        """
        eta = delta / (self.grad_rms(grad, eps=eps) + eps)
        return torch.clamp(eta, eta_min, eta_max)

    @torch.no_grad()
    def auto_lam_logit_grad_balance(
        self,
        a,
        g_meas,  # measurement-only grad wrt a
        mu_logit,
        alpha=0.1,  # target: RMS(g_meas) ≈ alpha * RMS(g_logit)
        lam_min=1e-4,
        lam_max=1e2,
        lam_prev=None,
        ema_beta=0.9,
        eps=1e-8,
    ):
        """
        All RMS values are computed over the whole batch, so lam is shared
        across samples (but still works fine for batched inputs).
        """
        b = a - mu_logit
        rms_meas = self.grad_rms(g_meas, eps=eps)
        rms_b = self.grad_rms(b, eps=eps)

        lam_raw = rms_meas / (alpha * (rms_b + eps) + eps)
        lam_raw = torch.clamp(lam_raw, lam_min, lam_max)

        if lam_prev is None:
            lam = lam_raw
        else:
            lam = ema_beta * lam_prev + (1.0 - ema_beta) * lam_raw
            lam = torch.clamp(lam, lam_min, lam_max)

        return lam, lam_raw, rms_meas, rms_b

    @torch.no_grad()
    def predict(self, dict_ord):
        """
        dict_ord["image"]: (B, C, H, W) — batch of images.
        Everything is kept fully batched throughout.
        """
        y = dict_ord["image"].to(self.device)

        device = y.device
        dtype = y.dtype
        B = y.shape[0]

        t_grid = torch.linspace(0.0, 0.99, steps=self.N, device=device)

        # init logits a (B, C, H, W) with scalar mu_logit
        a = torch.full_like(y, float(20.))

        # t=0
        t0_tensor = torch.full(
            (B, 1, 1, 1), float(t_grid[0]), device=device, dtype=dtype
        )
        sigma0 = self.inference_scheduler.time_steps(
            t0_tensor
        )  # expect broadcastable to y

        # ensure sigma0 is broadcastable to (B, C, H, W)
        while sigma0.dim() < y.dim():
            sigma0 = sigma0.unsqueeze(-1)

        x_t = sigma0 * torch.randn_like(y) + y

        for k, t_scalar in enumerate(t_grid):
            t_val = float(t_scalar)
            t_tensor = torch.full((B, 1, 1, 1), t_val, device=device, dtype=dtype)
            sigma = self.inference_scheduler.time_steps(t_tensor)

            # broadcast sigma to (B, C, H, W) if needed
            while sigma.dim() < y.dim():
                sigma = sigma.unsqueeze(-1)

            # prior mean from diffusion, fully batched
            x0_hat = self.model.sample_ode_from_xt(x_t, sigma, steps=self.ode_steps)

            x0 = 0.5 * (y.clone() + x0_hat.clone())

            lambda_k = torch.clamp(self.c_anneal * (sigma), min=5e-3)
            r_k = lambda_k

            for inner in range(self.N_inner):

                # --- x0 full step ---
                gx0 = self.grad_x0_energy(x0, x0_hat, a, y, r_k, lambda_k)

                eta_x = self.rms_normalized_stepsize(
                    gx0, delta=self.delta_x, eta_min=1e-5, eta_max=2e-2
                )

                x0 = self.langevin_update(
                    x0,
                    gx0,
                    eta=eta_x,
                    add_noise=True if k < self.N - 1 else False,
                    clip_grad=8e3,
                    clip_x=None,
                )

            if k < len(t_grid) - 1:
                t_next = float(t_grid[k + 1])
                t_next_tensor = torch.full(
                    (B, 1, 1, 1), t_next, device=device, dtype=dtype
                )
                sigma_next = self.inference_scheduler.time_steps(t_next_tensor)
                while sigma_next.dim() < y.dim():
                    sigma_next = sigma_next.unsqueeze(-1)
                x_t = x0 + sigma_next * torch.randn_like(x0)

        dict_pred = {
            "reconstruction": x0,  # (B, C, H, W)
            "anomaly_map": 1 - self.safe_sigmoid(a),
            "anomaly_map_2": (x0 - y).square(),
        }

        log.debug(f"[Predictor] Prediction complete for batch of size {y.shape[0]}")
        return dict_pred



class PredictorAblationB:
    def __init__(
        self,
        model: nn.Module,
        inference_scheduler: nn.Module,
        predictor_config,
    ):
        self._init_config(predictor_config)

        self.model = model
        self.model.eval()

        self.inference_scheduler = inference_scheduler

        # make sure everything is on the configured device
        self.to(self.device)

    def _init_config(self, predictor_config):
        self.N = predictor_config.N
        self.N_inner = predictor_config.N_inner

        self.lam_logit_init = predictor_config.lam_logit_init
        self.lam_smooth = predictor_config.lam_smooth
        self.alpha = predictor_config.alpha

        self.ode_steps = predictor_config.ode_steps
        self.c_anneal = predictor_config.c_anneal

        # device can be "cuda", "cuda:0", "cpu", or a torch.device
        dev = predictor_config.device
        self.device = torch.device(dev) if not isinstance(dev, torch.device) else dev

        self.mu_logit = predictor_config.mu_logit

        # these are referenced in predict(), so make sure they exist
        self.use_auto_lam_logit = getattr(predictor_config, "use_auto_lam_logit", True)
        self.delta_a = getattr(predictor_config, "delta_a", 1e-3)
        self.delta_x = getattr(predictor_config, "delta_x", 1e-3)

        self.eta_a_min = getattr(predictor_config, "eta_a_min", 1e-6)
        self.eta_a_max = getattr(predictor_config, "eta_a_max", 1e-2)

    def to(self, device):
        """Move model & scheduler to device and update self.device."""
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self.model.to(self.device)
        if hasattr(self.inference_scheduler, "to"):
            self.inference_scheduler.to(self.device)
        return self

    @torch.no_grad()
    def sigmoid_clamped(self, a, eps=1e-6):
        return torch.sigmoid(a).clamp(eps, 1 - eps)

    # optional alias to keep your original dict_pred line working
    @torch.no_grad()
    def safe_sigmoid(self, a, eps=1e-6):
        return self.sigmoid_clamped(a, eps=eps)

    @torch.no_grad()
    def grad_rms(self, x, eps=1e-8):
        # RMS over all elements (B, C, H, W)
        return torch.sqrt(torch.mean(x * x) + eps)

    @torch.no_grad()
    def grad_x0_energy(self, x0, x0_hat, a, y_k, sigma_prior, lambda_obs, eps=1e-8):
        """
        All inputs are batched tensors with the same shape as y:
        (B, C, H, W) after broadcasting.
        """
        M = self.sigmoid_clamped(a)
        inv_sig2 = 1.0 / (sigma_prior * sigma_prior + eps)
        inv_lam2 = 1.0 / (lambda_obs * lambda_obs + eps)
        grad_prior = (x0 - x0_hat) * inv_sig2
        grad_meas = (M * M) * (x0 - y_k) * inv_lam2
        return grad_prior + grad_meas

    @torch.no_grad()
    def grad_a_parts(
        self,
        x0,
        a,
        y_k,
        lambda_obs,
        lam_logit,
        lam_smooth,
        mu_logit,
        eps=1e-8,
    ):
        """
        All tensors are batched: (B, C, H, W) except scalars (lam_logit, lam_smooth, mu_logit).
        Broadcasting handles batch.
        Returns:
            grad_total, grad_meas, grad_logit, grad_smooth, grad_log
        """
        M = self.sigmoid_clamped(a)
        r = y_k - x0
        inv_lam2 = 1.0 / (lambda_obs * lambda_obs + eps)

        # measurement part
        grad_meas = M * M * (1.0 - M) * (r * r) * inv_lam2

        # barrier -sum log M (gradient wrt a)
        grad_log = -(1.0 - M)

        # quadratic prior in logits
        grad_logit = lam_logit * (a - mu_logit)

        # anisotropic smoothness
        grad_smooth = torch.zeros_like(a)
        if lam_smooth != 0.0:
            grad_smooth[:, :, :-1, :] += lam_smooth * (a[:, :, :-1, :] - a[:, :, 1:, :])
            grad_smooth[:, :, 1:, :] += lam_smooth * (a[:, :, 1:, :] - a[:, :, :-1, :])
            grad_smooth[:, :, :, :-1] += lam_smooth * (a[:, :, :, :-1] - a[:, :, :, 1:])
            grad_smooth[:, :, :, 1:] += lam_smooth * (a[:, :, :, 1:] - a[:, :, :, :-1])

        grad_total = grad_meas + grad_logit + grad_smooth + grad_log
        return grad_total, grad_meas, grad_logit, grad_smooth, grad_log

    @torch.no_grad()
    def langevin_update(
        self, x, grad, eta, add_noise=True, clip_grad=None, clip_x=None
    ):
        """
        x, grad: (B, C, H, W)
        eta: scalar or tensor broadcastable to x
        """
        if clip_grad is not None:
            grad = torch.clamp(grad, -clip_grad, clip_grad)

        eta = torch.as_tensor(eta, device=x.device, dtype=x.dtype)
        noise = torch.randn_like(x) if add_noise else 0.0
        x_new = x - eta * grad + torch.sqrt(2.0 * eta) * noise

        if clip_x is not None:
            x_new = torch.clamp(x_new, -clip_x, clip_x)
        return x_new

    @torch.no_grad()
    def rms_normalized_stepsize(self, grad, delta, eta_min, eta_max, eps=1e-8):
        """
        grad: (B, C, H, W)
        Returns a scalar eta shared across the batch
        (but still fully batch-compatible).
        """
        eta = delta / (self.grad_rms(grad, eps=eps) + eps)
        return torch.clamp(eta, eta_min, eta_max)

    @torch.no_grad()
    def auto_lam_logit_grad_balance(
        self,
        a,
        g_meas,  # measurement-only grad wrt a
        mu_logit,
        alpha=0.1,  # target: RMS(g_meas) ≈ alpha * RMS(g_logit)
        lam_min=1e-4,
        lam_max=1e2,
        lam_prev=None,
        ema_beta=0.9,
        eps=1e-8,
    ):
        """
        All RMS values are computed over the whole batch, so lam is shared
        across samples (but still works fine for batched inputs).
        """
        b = a - mu_logit
        rms_meas = self.grad_rms(g_meas, eps=eps)
        rms_b = self.grad_rms(b, eps=eps)

        lam_raw = rms_meas / (alpha * (rms_b + eps) + eps)
        lam_raw = torch.clamp(lam_raw, lam_min, lam_max)

        if lam_prev is None:
            lam = lam_raw
        else:
            lam = ema_beta * lam_prev + (1.0 - ema_beta) * lam_raw
            lam = torch.clamp(lam, lam_min, lam_max)

        return lam, lam_raw, rms_meas, rms_b

    @torch.no_grad()
    def predict(self, dict_ord):
        """
        dict_ord["image"]: (B, C, H, W) — batch of images.
        Everything is kept fully batched throughout.
        """
        y = dict_ord["image"].to(self.device)

        device = y.device
        dtype = y.dtype
        B = y.shape[0]

        t_grid = torch.linspace(0.0, 0.99, steps=self.N, device=device)

        # init logits a (B, C, H, W) with scalar mu_logit
        a = torch.full_like(y, float(self.mu_logit))

        # t=0
        t0_tensor = torch.full(
            (B, 1, 1, 1), float(t_grid[0]), device=device, dtype=dtype
        )
        sigma0 = self.inference_scheduler.time_steps(
            t0_tensor
        )  # expect broadcastable to y

        # ensure sigma0 is broadcastable to (B, C, H, W)
        while sigma0.dim() < y.dim():
            sigma0 = sigma0.unsqueeze(-1)

        x_t = sigma0 * torch.randn_like(y) + y

        lam_logit = torch.as_tensor(self.lam_logit_init, device=device, dtype=dtype)
        lam_logit_ema = lam_logit.clone()
        lambda_floor = 5e-3

        for k, t_scalar in enumerate(t_grid):
            t_val = float(t_scalar)
            t_tensor = torch.full((B, 1, 1, 1), t_val, device=device, dtype=dtype)
            sigma = self.inference_scheduler.time_steps(t_tensor)

            # broadcast sigma to (B, C, H, W) if needed
            while sigma.dim() < y.dim():
                sigma = sigma.unsqueeze(-1)

            # prior mean from diffusion, fully batched
            x0_hat = self.model.sample_ode_from_xt(x_t, sigma, steps=self.ode_steps)

            x0 = 0.5 * (y.clone() + x0_hat.clone())
        

            lambda_k = torch.full_like(sigma, 5e-2)
            r_k = lambda_k

            grad_total, _, _, _, g_log = self.grad_a_parts(
                x0=x0,
                a=a,
                y_k=y,
                lambda_obs=torch.clamp(lambda_k, 5e-2),
                lam_logit=torch.as_tensor(0.0, device=device, dtype=dtype),
                lam_smooth=0.0,
                mu_logit=self.mu_logit,
            )

            g_meas_base = grad_total - g_log

            if self.use_auto_lam_logit:
                lam_logit_ema, lam_logit_raw, rms_meas, rms_b = (
                    self.auto_lam_logit_grad_balance(
                        a=a,
                        g_meas=g_meas_base,
                        mu_logit=self.mu_logit,
                        alpha=self.alpha,
                        lam_min=1e-3,
                        lam_max=1e2,
                        lam_prev=lam_logit_ema,
                        ema_beta=0.95,
                    )
                )
                lam_logit = lam_logit_ema

            for inner in range(self.N_inner):
                # 1) measurement-only grad wrt a (for auto lam tuning)

                # 2) full gradient with current lam_logit and lam_smooth
                ga, g_meas, g_logit, g_smooth, g_log = self.grad_a_parts(
                    x0=x0,
                    a=a,
                    y_k=y,
                    lambda_obs=torch.clamp(lambda_k, 1e-2),
                    lam_logit=lam_logit,
                    lam_smooth=self.lam_smooth,
                    mu_logit=self.mu_logit,
                )

              
                eta_a = self.rms_normalized_stepsize(
                    ga,
                    delta=self.delta_a,
                    eta_min=self.eta_a_min,
                    eta_max=self.eta_a_max,
                )

                a = self.langevin_update(
                    a,
                    ga,
                    eta=eta_a,
                    add_noise=True,
                    clip_grad=50,
                    clip_x=None,
                )

                # --- x0 full step ---
                gx0 = self.grad_x0_energy(x0, x0_hat, a, y, r_k, lambda_k)

                eta_x = self.rms_normalized_stepsize(
                    gx0, delta=self.delta_x, eta_min=1e-5, eta_max=2e-2
                )

                x0 = self.langevin_update(
                    x0,
                    gx0,
                    eta=eta_x,
                    add_noise=True if k < self.N - 1 else False,
                    clip_grad=8e3,
                    clip_x=None,
                )

            if k < len(t_grid) - 1:
                t_next = float(t_grid[k + 1])
                t_next_tensor = torch.full(
                    (B, 1, 1, 1), t_next, device=device, dtype=dtype
                )
                sigma_next = self.inference_scheduler.time_steps(t_next_tensor)
                while sigma_next.dim() < y.dim():
                    sigma_next = sigma_next.unsqueeze(-1)
                x_t = x0 + sigma_next * torch.randn_like(x0)

        dict_pred = {
            "reconstruction": x0,  # (B, C, H, W)
            "anomaly_map": 1 - self.safe_sigmoid(a),
            "anomaly_map_2": (x0 - y).square(),
        }

        log.debug(f"[Predictor] Prediction complete for batch of size {y.shape[0]}")
        return dict_pred


class PredictorAblationC:
    def __init__(
        self,
        model: nn.Module,
        inference_scheduler: nn.Module,
        predictor_config,
    ):
        self._init_config(predictor_config)

        self.model = model
        self.model.eval()

        self.inference_scheduler = inference_scheduler

        # make sure everything is on the configured device
        self.to(self.device)

    def _init_config(self, predictor_config):
        self.N = predictor_config.N
        self.N_inner = predictor_config.N_inner

        self.lam_logit_init = predictor_config.lam_logit_init
        self.lam_smooth = predictor_config.lam_smooth
        self.alpha = predictor_config.alpha

        self.ode_steps = predictor_config.ode_steps
        self.c_anneal = predictor_config.c_anneal

        # device can be "cuda", "cuda:0", "cpu", or a torch.device
        dev = predictor_config.device
        self.device = torch.device(dev) if not isinstance(dev, torch.device) else dev

        self.mu_logit = predictor_config.mu_logit

        # these are referenced in predict(), so make sure they exist
        self.use_auto_lam_logit = getattr(predictor_config, "use_auto_lam_logit", True)
        self.delta_a = getattr(predictor_config, "delta_a", 1e-3)
        self.delta_x = getattr(predictor_config, "delta_x", 1e-3)

        self.eta_a_min = getattr(predictor_config, "eta_a_min", 1e-6)
        self.eta_a_max = getattr(predictor_config, "eta_a_max", 1e-2)

    def to(self, device):
        """Move model & scheduler to device and update self.device."""
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self.model.to(self.device)
        if hasattr(self.inference_scheduler, "to"):
            self.inference_scheduler.to(self.device)
        return self

    @torch.no_grad()
    def sigmoid_clamped(self, a, eps=1e-6):
        return torch.sigmoid(a).clamp(eps, 1 - eps)

    # optional alias to keep your original dict_pred line working
    @torch.no_grad()
    def safe_sigmoid(self, a, eps=1e-6):
        return self.sigmoid_clamped(a, eps=eps)

    @torch.no_grad()
    def grad_rms(self, x, eps=1e-8):
        # RMS over all elements (B, C, H, W)
        return torch.sqrt(torch.mean(x * x) + eps)

    @torch.no_grad()
    def grad_x0_energy(self, x0, x0_hat, a, y_k, sigma_prior, lambda_obs, eps=1e-8):
        """
        All inputs are batched tensors with the same shape as y:
        (B, C, H, W) after broadcasting.
        """
        M = self.sigmoid_clamped(a)
        inv_sig2 = 1.0 / (sigma_prior * sigma_prior + eps)
        inv_lam2 = 1.0 / (lambda_obs * lambda_obs + eps)
        grad_prior = (x0 - x0_hat) * inv_sig2
        grad_meas = (M * M) * (x0 - y_k) * inv_lam2
        return grad_prior + grad_meas

    @torch.no_grad()
    def grad_a_parts(
        self,
        x0,
        a,
        y_k,
        lambda_obs,
        lam_logit,
        lam_smooth,
        mu_logit,
        eps=1e-8,
    ):
        """
        All tensors are batched: (B, C, H, W) except scalars (lam_logit, lam_smooth, mu_logit).
        Broadcasting handles batch.
        Returns:
            grad_total, grad_meas, grad_logit, grad_smooth, grad_log
        """
        M = self.sigmoid_clamped(a)
        r = y_k - x0
        inv_lam2 = 1.0 / (lambda_obs * lambda_obs + eps)

        # measurement part
        grad_meas = M * M * (1.0 - M) * (r * r) * inv_lam2

        # barrier -sum log M (gradient wrt a)
        grad_log = -(1.0 - M)

        # quadratic prior in logits
        grad_logit = lam_logit * (a - mu_logit)

        # anisotropic smoothness
        grad_smooth = torch.zeros_like(a)
        if lam_smooth != 0.0:
            grad_smooth[:, :, :-1, :] += lam_smooth * (a[:, :, :-1, :] - a[:, :, 1:, :])
            grad_smooth[:, :, 1:, :] += lam_smooth * (a[:, :, 1:, :] - a[:, :, :-1, :])
            grad_smooth[:, :, :, :-1] += lam_smooth * (a[:, :, :, :-1] - a[:, :, :, 1:])
            grad_smooth[:, :, :, 1:] += lam_smooth * (a[:, :, :, 1:] - a[:, :, :, :-1])

        grad_total = grad_meas + grad_logit + grad_smooth + grad_log
        return grad_total, grad_meas, grad_logit, grad_smooth, grad_log

    @torch.no_grad()
    def langevin_update(
        self, x, grad, eta, add_noise=True, clip_grad=None, clip_x=None
    ):
        """
        x, grad: (B, C, H, W)
        eta: scalar or tensor broadcastable to x
        """
        if clip_grad is not None:
            grad = torch.clamp(grad, -clip_grad, clip_grad)

        eta = torch.as_tensor(eta, device=x.device, dtype=x.dtype)
        noise = torch.randn_like(x) if add_noise else 0.0
        x_new = x - eta * grad + torch.sqrt(2.0 * eta) * noise

        if clip_x is not None:
            x_new = torch.clamp(x_new, -clip_x, clip_x)
        return x_new

    @torch.no_grad()
    def rms_normalized_stepsize(self, grad, delta, eta_min, eta_max, eps=1e-8):
        """
        grad: (B, C, H, W)
        Returns a scalar eta shared across the batch
        (but still fully batch-compatible).
        """
        eta = delta / (self.grad_rms(grad, eps=eps) + eps)
        return torch.clamp(eta, eta_min, eta_max)

    @torch.no_grad()
    def auto_lam_logit_grad_balance(
        self,
        a,
        g_meas,  # measurement-only grad wrt a
        mu_logit,
        alpha=0.1,  # target: RMS(g_meas) ≈ alpha * RMS(g_logit)
        lam_min=1e-4,
        lam_max=1e2,
        lam_prev=None,
        ema_beta=0.9,
        eps=1e-8,
    ):
        """
        All RMS values are computed over the whole batch, so lam is shared
        across samples (but still works fine for batched inputs).
        """
        b = a - mu_logit
        rms_meas = self.grad_rms(g_meas, eps=eps)
        rms_b = self.grad_rms(b, eps=eps)

        lam_raw = rms_meas / (alpha * (rms_b + eps) + eps)
        lam_raw = torch.clamp(lam_raw, lam_min, lam_max)

        if lam_prev is None:
            lam = lam_raw
        else:
            lam = ema_beta * lam_prev + (1.0 - ema_beta) * lam_raw
            lam = torch.clamp(lam, lam_min, lam_max)

        return lam, lam_raw, rms_meas, rms_b

    @torch.no_grad()
    def predict(self, dict_ord):
        """
        dict_ord["image"]: (B, C, H, W) — batch of images.
        Everything is kept fully batched throughout.
        """
        y = dict_ord["image"].to(self.device)

        device = y.device
        dtype = y.dtype
        B = y.shape[0]

        t_grid = torch.linspace(0.0, 0.99, steps=self.N, device=device)

        # init logits a (B, C, H, W) with scalar mu_logit
        a = torch.full_like(y, float(self.mu_logit))

        # t=0
        t0_tensor = torch.full(
            (B, 1, 1, 1), float(t_grid[0]), device=device, dtype=dtype
        )
        sigma0 = self.inference_scheduler.time_steps(
            t0_tensor
        )  # expect broadcastable to y

        # ensure sigma0 is broadcastable to (B, C, H, W)
        while sigma0.dim() < y.dim():
            sigma0 = sigma0.unsqueeze(-1)

        x_t = sigma0 * torch.randn_like(y) + y

        lam_logit = torch.as_tensor(self.lam_logit_init, device=device, dtype=dtype)
        lam_logit_ema = lam_logit.clone()
        lambda_floor = 5e-3

        for k, t_scalar in enumerate(t_grid):
            t_val = float(t_scalar)
            t_tensor = torch.full((B, 1, 1, 1), t_val, device=device, dtype=dtype)
            sigma = self.inference_scheduler.time_steps(t_tensor)

            # broadcast sigma to (B, C, H, W) if needed
            while sigma.dim() < y.dim():
                sigma = sigma.unsqueeze(-1)

            # prior mean from diffusion, fully batched
            x0_hat = self.model.sample_ode_from_xt(x_t, sigma, steps=self.ode_steps)

            x0 = 0.5 * (y.clone() + x0_hat.clone())


            lambda_k = torch.clamp(self.c_anneal * (sigma), min=lambda_floor)
            r_k = lambda_k

            grad_total, _, _, _, g_log = self.grad_a_parts(
                x0=x0,
                a=a,
                y_k=y,
                lambda_obs=torch.clamp(lambda_k, 5e-2),
                lam_logit=torch.as_tensor(0.0, device=device, dtype=dtype),
                lam_smooth=0.0,
                mu_logit=self.mu_logit,
            )

            g_meas_base = grad_total - g_log

            if self.use_auto_lam_logit:
                lam_logit_ema, lam_logit_raw, rms_meas, rms_b = (
                    self.auto_lam_logit_grad_balance(
                        a=a,
                        g_meas=g_meas_base,
                        mu_logit=self.mu_logit,
                        alpha=self.alpha,
                        lam_min=1e-3,
                        lam_max=1e2,
                        lam_prev=lam_logit_ema,
                        ema_beta=0.95,
                    )
                )
                lam_logit = lam_logit_ema

            for inner in range(self.N_inner):
                # 1) measurement-only grad wrt a (for auto lam tuning)

                # 2) full gradient with current lam_logit and lam_smooth
                ga, g_meas, g_logit, g_smooth, g_log = self.grad_a_parts(
                    x0=x0,
                    a=a,
                    y_k=y,
                    lambda_obs=torch.clamp(lambda_k, 5e-2),
                    lam_logit=lam_logit,
                    lam_smooth=self.lam_smooth,
                    mu_logit=self.mu_logit,
                )

               
                eta_a = self.rms_normalized_stepsize(
                    ga,
                    delta=self.delta_a,
                    eta_min=self.eta_a_min,
                    eta_max=self.eta_a_max,
                )

                a = self.langevin_update(
                    a,
                    ga,
                    eta=eta_a,
                    add_noise=True,
                    clip_grad=50,
                    clip_x=None,
                )

                # --- x0 full step ---
                gx0 = self.grad_x0_energy(x0, x0_hat, a, y, r_k, lambda_k)

                eta_x = self.rms_normalized_stepsize(
                    gx0, delta=self.delta_x, eta_min=1e-5, eta_max=2e-2
                )

                x0 = self.langevin_update(
                    x0,
                    gx0,
                    eta=eta_x,
                    add_noise=True if k < self.N - 1 else False,
                    clip_grad=8e3,
                    clip_x=None,
                )

            if k < len(t_grid) - 1:
                t_next = float(t_grid[k + 1])
                t_next_tensor = torch.full(
                    (B, 1, 1, 1), t_next, device=device, dtype=dtype
                )
                sigma_next = self.inference_scheduler.time_steps(t_next_tensor)
                while sigma_next.dim() < y.dim():
                    sigma_next = sigma_next.unsqueeze(-1)
                x_t = x0 + sigma_next * torch.randn_like(x0)

        dict_pred = {
            "reconstruction": x0,  # (B, C, H, W)
            "anomaly_map": 1 - self.safe_sigmoid(a),
            "anomaly_map_2": (x0 - y).square(),
        }

        log.debug(f"[Predictor] Prediction complete for batch of size {y.shape[0]}")
        return dict_pred
