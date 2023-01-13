import numpy as np
import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    """
    General Gaussian diffusion module

    Args:
        schedule (`beta_schedule.BaseSchedule`): beta schedule instance.
        time_aware_net (`nn.Module`): time aware neural network instance.
        reverse_variance (`str`, *optional*, defaults to 'beta'): variance for reverse process, can be selected from
            ['beta', 'sigma_bar_square', 'learnable'].
        prediction (`str`, *optional*, defaults to 'reconstruction'): what the time_aware_net aims to predict from the
            input, can be selected from ['direct', 'reconstruction', 'denoising'].
        loss_mode (`str`, *optional*, defaults to 'simple'): can be selected from ['full', 'simple'].
    """

    def __init__(self,
                 schedule,
                 time_aware_net,
                 reverse_variance='beta',
                 prediction='reconstruction',
                 loss_mode='simple'):
        super().__init__()
        self.schedule = schedule
        self.T = len(self.schedule)
        self.time_aware_net = time_aware_net
        self.sigma_bar_square = np.zeros(len(self.schedule))
        self.sigma_bar_square[0] = self.schedule.beta[0]
        self.sigma_bar_square[1:] = self.schedule.beta[1:] * (1 - self.schedule.alpha_bar[:-1]) \
                                    / (1 - self.schedule.alpha_bar[1:])

        # set variance for reverse process
        if reverse_variance == 'beta':
            self.sigma_square = self.schedule.beta
        elif reverse_variance == 'sigma_bar_square':
            self.sigma_square = self.sigma_bar_square
        elif reverse_variance == 'learnable':
            self.sigma_square = nn.Parameter(torch.tensor(self.sigma_bar_square))
        else:
            raise ValueError('Please specify reverse_variance to beta, sigma_bar_square or learnable!')
        self.reverse_variance = reverse_variance

        # set prediction mode
        if prediction not in ['direct', 'reconstruction', 'denoising']:
            raise ValueError('Please specify prediction to direct, reconstruction or denoising!')
        self.prediction = prediction

        # set loss mode, if calculate the full version, then coefficients for weights are recorded
        if loss_mode == 'full':
            self.loss_fn = self.loss_full
            self.loss_rec_coef = torch.tensor(self.schedule.alpha_bar[:-1] * self.schedule.beta[1:] ** 2
                                              / (1 - self.schedule.alpha_bar[1:]) ** 2)
            self.loss_rec_coef = torch.cat([torch.tensor([1]), self.loss_rec_coef])
            self.loss_den_coef = torch.tensor(
                self.schedule.beta ** 2 / (self.schedule.alpha * (1 - self.schedule.alpha_bar)))
        elif loss_mode == 'simple':
            self.loss_fn = self.loss_simple
        else:
            raise ValueError('Please specify loss_mode to full or simple!')
        self.loss_mode = loss_mode

        # record coefficients for distributions
        self.coef_q_xt_given_xt_minus_1 = (torch.tensor(np.sqrt(1 - self.schedule.beta)),
                                           torch.tensor(np.sqrt(self.schedule.beta)))
        self.coef_q_xt_given_x0 = (torch.tensor(np.sqrt(self.schedule.alpha_bar)),
                                   torch.tensor(np.sqrt(1 - self.schedule.alpha_bar)))

        # record coefficients for calculating mu_bar or mu
        self.coef_reconstruction = [
            torch.tensor((1 - self.schedule.alpha_bar[:-1]) * np.sqrt(self.schedule.alpha[1:]) / (
                    1 - self.schedule.alpha_bar[1:])),
            torch.tensor(
                self.schedule.beta[1:] * np.sqrt(self.schedule.alpha_bar[:-1]) / (1 - self.schedule.alpha_bar[1:]))]
        self.coef_reconstruction[0] = torch.cat([torch.tensor([0]), self.coef_reconstruction[0]])
        self.coef_reconstruction[1] = torch.cat([torch.tensor([1]), self.coef_reconstruction[1]])
        self.coef_denoising = (torch.tensor(1 / np.sqrt(self.schedule.alpha)),
                               torch.tensor(self.schedule.beta / np.sqrt(
                                   (1 - self.schedule.alpha_bar) * self.schedule.alpha)))

    def sample_q_xt_given_xt_minus_1(self, xt_minus_1, t):
        dtype = xt_minus_1.dtype
        device = xt_minus_1.device
        epsilon = torch.randn(xt_minus_1.shape, device=device)
        mu = self.coef_q_xt_given_xt_minus_1[0][t - 1].to(dtype).to(device).view(-1, *([1] * (len(epsilon.shape) - 1)))
        sigma = self.coef_q_xt_given_xt_minus_1[1][t - 1].to(dtype).to(device).view(*mu.shape)
        return mu * xt_minus_1 + sigma * epsilon, epsilon

    def sample_q_xt_given_x0(self, x0, t):
        dtype = x0.dtype
        device = x0.device
        epsilon = torch.randn(x0.shape, device=device)
        mu = self.coef_q_xt_given_x0[0][t - 1].to(dtype).to(device).view(-1, *([1] * (len(epsilon.shape) - 1)))
        sigma = self.coef_q_xt_given_x0[1][t - 1].to(dtype).to(device).view(*mu.shape)
        return mu * x0 + sigma * epsilon, epsilon

    def sample_p_xT(self, shape, device):
        epsilon = torch.randn(shape, device=device)
        return epsilon

    def sample_p_xt_minus_1_given_xt(self, xt, t):
        dtype = xt.dtype
        device = xt.device
        epsilon = torch.randn(xt.shape, device=device)
        if self.prediction == 'direct':
            mu = self.time_aware_net(xt, t)
        elif self.prediction == 'reconstruction':
            x0_hat = self.time_aware_net(xt, t)
            coef1 = self.coef_reconstruction[0][t - 1].to(dtype).to(device).view(-1, *([1] * (len(xt.shape) - 1)))
            coef2 = self.coef_reconstruction[1][t - 1].to(dtype).to(device).view(*coef1.shape)
            mu = coef1 * xt + coef2 * x0_hat
        else:
            e0_hat = self.time_aware_net(xt, t)
            coef1 = self.coef_denoising[0][t - 1].to(dtype).to(device).view(-1, *([1] * (len(xt.shape) - 1)))
            coef2 = self.coef_denoising[1][t - 1].to(dtype).to(device).view(*coef1.shape)
            mu = coef1 * xt - coef2 * e0_hat
        sigma = torch.sqrt(torch.tensor(
            self.sigma_square[t - 1])).to(dtype).to(device).view(-1, *([1] * (len(xt.shape) - 1)))
        return mu + sigma * epsilon

    def sub_direct_loss(self, xt, x0, t):
        B = xt.shape[0]
        coef1 = self.coef_reconstruction[0][t - 1].to(xt.dtype).to(xt.device).view(-1, *([1] * (len(xt.shape) - 1)))
        coef2 = self.coef_reconstruction[1][t - 1].to(xt.dtype).to(xt.device).view(-1, *([1] * (len(xt.shape) - 1)))
        mu_bar = coef1 * xt + coef2 * x0
        mu_hat = self.time_aware_net(xt, t)
        return torch.sum(torch.square(mu_hat.view(B, -1) - mu_bar.view(B, -1)), dim=1)

    def sub_reconstruction_loss(self, xt, x0, t):
        B = xt.shape[0]
        x0_hat = self.time_aware_net(xt, t)
        return torch.sum(torch.square(x0_hat.view(B, -1) - x0.view(B, -1)), dim=1)

    def sub_denoising_loss(self, xt, et, t):
        B = xt.shape[0]
        et_hat = self.time_aware_net(xt, t)
        return torch.sum(torch.square(et_hat.view(B, -1) - et.view(B, -1)), dim=1)

    def loss_full(self, x0):
        loss = 0
        for t in range(1, self.T + 1):
            t = torch.tensor([t])
            xt, et = self.sample_q_xt_given_x0(x0, t)
            if self.prediction == 'direct':
                weight = 1 / self.sigma_square[t - 1]
                loss += self.sub_direct_loss(xt, x0, t) * weight
            elif self.prediction == 'reconstruction':
                weight = self.loss_rec_coef[t - 1].to(x0.dtype).to(x0.device) / self.sigma_square[t - 1]
                loss += self.sub_reconstruction_loss(xt, x0, t) * weight
            else:
                weight = self.loss_den_coef[t - 1].to(x0.dtype).to(x0.device) / self.sigma_square[t - 1]
                loss += self.sub_denoising_loss(xt, et, t) * weight

        if self.reverse_variance == 'learnable':
            n = sum(x0.shape[1:])
            loss = loss / n
            loss += torch.sum(torch.log(self.sigma_square).to(loss.device))
            loss += torch.sum((torch.tensor(self.sigma_bar_square).to(loss.device) / self.sigma_square)[1:])
        return loss

    def loss_simple(self, x0):
        t = torch.randint(self.T, (x0.shape[0],)) + 1
        xt, et = self.sample_q_xt_given_x0(x0, t)
        if self.prediction == 'direct':
            loss = self.sub_direct_loss(xt, x0, t)
        elif self.prediction == 'reconstruction':
            loss = self.sub_reconstruction_loss(xt, x0, t)
        else:
            loss = self.sub_denoising_loss(xt, et, t)
        return loss

    def forward(self, x0):
        loss = self.loss_fn(x0)
        return torch.mean(loss)

    def generate(self, x, t_start, t_end):
        assert 0 <= t_end < t_start <= self.T
        with torch.no_grad():
            xt = x.clone()
            for i in range(t_start, t_end, -1):
                xt_minus_1 = self.sample_p_xt_minus_1_given_xt(xt, torch.tensor([i]))
                xt = xt_minus_1
            return xt


if __name__ == '__main__':
    print('dummy test')
    from beta_schedule import LinearSchedule
    from time_aware_module import TimeAwareUNet
    from torch.optim import Adam
    import time

    lr = 2e-4
    device = 'cuda:0'
    timesteps = 1000
    torch.manual_seed(0)

    x0 = torch.randn(2, 3, 32, 32).to(device)
    schedule = LinearSchedule(timesteps=timesteps)
    time_aware_net = TimeAwareUNet(3, timesteps)
    for r in ['beta', 'sigma_bar_square', 'learnable']:
        for p in ['direct', 'reconstruction', 'denoising']:
            for l in ['full', 'simple']:
                torch.cuda.reset_peak_memory_stats()
                diffusion_model = GaussianDiffusion(schedule, time_aware_net, r, p, l)
                diffusion_model.to(device)
                optimizer = Adam(diffusion_model.parameters(), lr=lr)
                start = time.time()
                optimizer.zero_grad()
                loss = diffusion_model(x0)
                loss.backward()
                optimizer.step()
                print(f'{r}, {p}, {l}, {loss.item():.4f}, {time.time() - start:.4f}s, '
                      f'{torch.cuda.max_memory_allocated() / 2 ** 20:.2f}MB')
