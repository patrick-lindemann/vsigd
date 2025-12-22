import numpy as np
import torch
import theseus as th
from theseus import SO3

from lib_nn.math_utils import SO3_R3


class AnnLangDynSE3:
    """
    Annealed Langevin Dynamics for SE(3)
    Adapted from: SE(3) Diffusion Fields
    """

    def __init__(
        self,
        model,
        T=200,
        k_steps=1,
        T_fit=5,
        deterministic=False,
        sdf_cost=False,
        sdf_grad_align=False,
        device="cuda:0",
    ):

        self.model = model
        self.device = device
        self.shape = [4, 4]

        ## Langevin Dynamics evolution ##
        self.T = T
        self.T_fit = T_fit
        self.k_steps = k_steps
        self.deterministic = deterministic
        self.sdf_cost = sdf_cost
        self.sdf_grad_align = sdf_grad_align

    @staticmethod
    def _marginal_prob_std(t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.0) / (2.0 * np.log(sigma)))

    def _step(self, H0, t, noise_off=True, c_lr_decay=None):

        ## Phase
        noise_std = 0.5
        eps = 1e-3
        phase = ((self.T - t) / (self.T)) + eps
        sigma_T = self._marginal_prob_std(eps)

        ## Annealed Langevin Dynamics ##
        alpha = 1e-3
        sigma_i = self._marginal_prob_std(phase)
        ratio = sigma_i**2 / sigma_T**2
        c_lr = alpha * ratio
        if noise_off:
            c_lr = 0.003
        if c_lr_decay is not None:
            c_lr = c_lr * c_lr_decay

        H1 = H0
        for k in range(self.k_steps):
            ## 1.Set input variable to Theseus ##
            H0_in = SO3_R3(R=H1[:, :3, :3], t=H1[:, :3, -1])
            phi0 = H0_in.log_map()

            ## 2. Compute energy gradient ##
            phi0_in = phi0.detach().requires_grad_(True)
            H_in = SO3_R3().exp_map(phi0_in).to_matrix()
            t_in = phase * torch.ones_like(H_in[:, 0, 0])
            e = self.model(H_in, t_in)
            d_phi = torch.autograd.grad(e.sum(), phi0_in)[0]

            ## 3. Compute noise vector ##
            if noise_off:
                noise = torch.zeros_like(phi0_in)
            else:
                noise = torch.randn_like(phi0_in) * noise_std

            ## 4. Compute translation ##
            delta = -c_lr / 2 * d_phi + np.sqrt(c_lr) * noise

            t_delta = delta[:, :3]
            w_Delta = SO3().exp_map(delta[:, 3:])

            ## 5. Move the points ##
            R1_out = th.compose(w_Delta, H0_in.R)
            t1_out = H0_in.t + t_delta
            H1 = SO3_R3(R=R1_out, t=t1_out).to_matrix()

        return H1

    def sample(self, batch_size, get_traj=False):
        ## 1.Sample initial SE(3) ##
        H0 = SO3_R3().sample(batch_size).to(self.device, torch.float32)

        ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
        Ht = H0
        if get_traj:
            trj_H = Ht[None, ...]

        # inverse diffusion steps
        for t in range(self.T):
            Ht = self._step(Ht, t, noise_off=self.deterministic)
            if get_traj:
                trj_H = torch.cat((trj_H, Ht[None, :]), 0)

        # final fitting steps
        for t in range(self.T_fit):
            Ht = self._step(Ht, self.T, noise_off=True)
            if get_traj:
                trj_H = torch.cat((trj_H, Ht[None, :]), 0)

        if get_traj:
            return Ht, trj_H
        return Ht
