import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionTrajectoryModel(nn.Module):
    def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

        # beta schedule
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    # x_0 -> x_t (Noisy trajectory)
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus * noise
        return x_t, noise

    # Training step: predict noise from x_t
    def forward(self, x_0, cond_info=None):
        B = x_0.size(0)
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        x_t, noise = self.q_sample(x_0, t)
        x_t = x_t.permute(0, 3, 2, 1)
        noise = noise.permute(0, 3, 2, 1)
        
        if cond_info is not None:
            cond_info = cond_info.to(self.device)

        noise_pred = self.model(x_t, t, cond_info)
        noise_loss = F.mse_loss(noise_pred, noise)
        
        alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        x_0_pred = (x_t - torch.sqrt(1.0 - alpha_hat) * noise_pred) / torch.sqrt(alpha_hat)
        x_0_pred = x_0_pred.permute(0, 3, 2, 1)  # [B, T, 11, 2]
        x_0 = x_0.to(self.device)
        
        # player-wise loss
        player_loss = F.mse_loss(x_0_pred, x_0, reduction='none')  # [B, T, 11, 2]
        player_loss = player_loss.mean(dim=[1, 3]).mean()
        return noise_loss, player_loss


    # Sampling (generate trajectory)
    @torch.no_grad()
    def generate(self, shape, cond_info=None, num_samples=10):
        B, T, N_players, D = shape  # shape = [B, T, 11, 2]
        cond_info = cond_info.to(self.device)
        cond_info = cond_info.repeat(num_samples, 1, 1, 1)  # [N*B, F, 11, T]

        x_t = torch.randn((num_samples * B, T, N_players, D), device=self.device)
        x_t = x_t.permute(0, 3, 2, 1)  # [N*B, 2, 11, T]

        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((num_samples * B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x_t, t_tensor, cond_info)

            alpha = self.alphas[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.betas[t]

            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            x_t = (1 / torch.sqrt(alpha)) * (
                x_t - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred
            ) + torch.sqrt(beta) * noise

        x_t = x_t.permute(0, 3, 2, 1)  # [N*B, T, 11, 2]
        x_t = x_t.view(num_samples, B, T, N_players, D)  # [N, B, T, 11, 2]
        return x_t

