import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

from lib_nn.math_utils import SO3_R3


def loss_sdf_gdiff(sdf_pred, sdf_gt, clip_delta=0.6):
    """
    SDF reconstruction loss (L1 with clipping)
    assumes model.set_latent(c) has been called
    - c = model_input['visual_context']
    - x_sdf = model_input['x_sdf'].detach().requires_grad_()
    - sdf_pred = model.compute_sdf(x_sdf.view(-1, 3))
    """
    pred_clip_sdf = torch.clip(sdf_pred, -10.0, clip_delta)
    target_clip_sdf = torch.clip(sdf_gt, -10.0, clip_delta)
    l_rec = F.l1_loss(pred_clip_sdf, target_clip_sdf, reduction="mean")
    return l_rec


def pre_loss_projected_denoising_gdiff(H, model, sigma=0.5, eps=1e-5):
    """
    H: B x 4 x 4
    sigma: scalar
    """
    ## 1. H to vector ##
    H_th = SO3_R3(R=H[..., :3, :3], t=H[..., :3, -1])
    xw = H_th.log_map()

    ## 2. Sample perturbed datapoint ##
    random_t = torch.rand_like(xw[..., 0], device=xw.device) * (1.0 - eps) + eps
    z = torch.randn_like(xw)
    # marginal probability std
    std = torch.sqrt((sigma ** (2 * random_t) - 1.0) / (2.0 * np.log(sigma)))
    perturbed_x = xw + z * std[..., None]
    perturbed_x = perturbed_x.detach()
    perturbed_x.requires_grad_(True)

    ## Get gradient ##
    with torch.set_grad_enabled(True):
        perturbed_H = SO3_R3().exp_map(perturbed_x).to_matrix()
        energy = model(perturbed_H, random_t)
        grad_energy = torch.autograd.grad(
            energy.sum(),
            perturbed_x,
            only_inputs=True,
            retain_graph=True,
            create_graph=True,
        )[0]
    # target
    z_target = z / std[..., None]
    return grad_energy, z_target


def loss_projected_denoising_gdiff(grad_energy, z_target):
    """
    SE(3) denoising loss for diffusion model (L1 loss)
    assumes:
    - model.set_latent(c) has been called
    - pre_loss_projected_denoising_gdiff is used for:
        - grad_energy
        - z_target
    """
    # Compute L1 loss
    loss = F.l1_loss(grad_energy, z_target, reduction="mean") / 10.0
    return loss
