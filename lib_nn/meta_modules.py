from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from lib_nn.lib_pointvae import conv_pointnet
from lib_nn.lib_pointvae import autoencoder
from lib_nn.feature_net import TimeLatentFeatureEncoder
from lib_nn.langevin_dynamics import AnnLangDynSE3
from lib_nn import losses
from lib_nn import torch_utils


def map_projected_points(H, p):
    p_ext = torch.cat((p, torch.ones_like(p[..., :1])), -1)
    p_alig = torch.einsum("...md,pd->...pm", H, p_ext)[..., :-1]
    return p_alig


class MetaVSIGD(nn.Module):
    def __init__(self, net_cfg, opt_cfg):
        super().__init__()
        self.net_cfg = net_cfg
        self.opt_cfg = opt_cfg
        self.loss_cfg = opt_cfg["loss"]

        self.load_network()
        self.unset_precompute()

    def load_network(self):
        # ----------------- PointVAE -----------------
        # sdf net
        cfg_ptvae = self.net_cfg["ptvae"]

        # pc features autoencoder
        feature_dim = cfg_ptvae["latent_dim"]  # latent dim of pointnet
        modulation_dim = feature_dim * 3  # latent dim of modulation
        latent_std = cfg_ptvae.get("latent_std", 0.25)  # for latent space gaussian
        conv_out_size = cfg_ptvae.get("conv_out_size", (2, 2))
        hidden_dims = [
            modulation_dim,
            modulation_dim,
            modulation_dim,
            modulation_dim,
            modulation_dim,
        ]
        self.vae_model = autoencoder.BetaVAE(
            in_channels=feature_dim * 3,
            latent_dim=modulation_dim,
            hidden_dims=hidden_dims,
            kl_std=latent_std,
            conv_out_size=conv_out_size,
        )
        self.pointnet = conv_pointnet.ConvPointnet(
            c_dim=feature_dim,
            hidden_dim=cfg_ptvae.get("pn_hidden_dim", feature_dim),
            plane_resolution=cfg_ptvae.get("plane_resolution", 64),
        )

        # load pretrained shape encoder
        ptnet_state_dict = torch.load(cfg_ptvae["ptnet_ckpt"])
        bvae_model_state_dict = torch.load(cfg_ptvae["bvae_ckpt"])
        self.vae_model.load_state_dict(bvae_model_state_dict)
        self.pointnet.load_state_dict(ptnet_state_dict)

        # ----------------- Grasp Diffusion -----------------
        # pre-sampled gripper points
        cfg_points = self.net_cfg["points"]
        self.n_gripper_pts = cfg_points["n_points"]
        loc = np.array(cfg_points["loc"], dtype=np.float32)
        scale = np.array(cfg_points["scale"], dtype=np.float32)

        np.random.seed(seed=123)
        points = np.random.rand(1000, 3).astype(np.float32)
        points = (points - 0.5) * 2.0
        points = points[: self.n_gripper_pts, :] * scale + loc
        points_gripper = torch.from_numpy(points).float()  # / 8.0
        self.register_buffer("points_gripper", points_gripper)

        # feature net
        cfg_feat_enc = self.net_cfg["feat_net"]
        latent_size = cfg_feat_enc["latent_size_point_features"]

        self.feat_net = TimeLatentFeatureEncoder(
            dims=cfg_feat_enc["dims"],
            in_dim=cfg_feat_enc.get("in_dim", 3),
            enc_dim=cfg_feat_enc["enc_dim"],
            out_dim=cfg_feat_enc.get("out_dim", 1),
            activation=cfg_feat_enc.get("activation", "relu"),
            latent_size=latent_size,
            latent_in=cfg_feat_enc["latent_in"],
            cond_fn_type_in=cfg_feat_enc.get("cond_fn_in", "cc"),  # cc or film
            cond_fn_type=cfg_feat_enc.get("cond_fn", "cc"),  # cc or film
            dropout=cfg_feat_enc.get("dropout", None),
        )

        # Energy Based Model
        cfg_energy = self.net_cfg["energy_net"]
        in_dim = cfg_points["n_points"] * cfg_feat_enc["out_dim"]
        hidden_dim = cfg_energy["hidden_dim"]
        self.energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # default mode
        self.pointnet.eval()
        self.vae_model.eval()
        self.feat_net.eval()
        self.energy_net.eval()

    def get_opt_params(self):
        lrs = self.opt_cfg["lr_per_group"]
        wds = self.opt_cfg.get("wd_per_group", None)
        if wds is None:
            wds = {
                "feat_net": 0.0,
                "energy_net": 0.0,
            }

        opt_params = []

        # sdf and vae
        # sdf - vae
        if lrs.get("vae_model", None) is None or lrs["vae_model"] < 0.0:
            torch_utils.freeze_model_(self.vae_model)
            print(f"> Frozen VAE model")
        elif lrs["vae_model"] == 0.0:
            print(f"> Not optimizing VAE model")
            # do not add to opt_params
        else:
            opt_params.append(
                {
                    "params": self.vae_model.parameters(),
                    "lr": lrs["vae_model"],
                    "weight_decay": wds.get("vae_model", 0.0),
                }
            )
            print(f"> Optimizing VAE model")

        # sdf - pointnet
        if lrs.get("pointnet", None) is None or lrs["pointnet"] < 0.0:
            torch_utils.freeze_model_(self.pointnet)
            print(f"> Frozen Pointnet")
        elif lrs["pointnet"] == 0.0:
            print(f"> Not optimizing Pointnet")
            # do not add to opt_params
        else:
            opt_params.append(
                {
                    "params": self.pointnet.parameters(),
                    "lr": lrs["pointnet"],
                    "weight_decay": wds.get("pointnet", 0.0),
                }
            )
            print(f"> Optimizing Pointnet")

        # feature net
        params_feat_decay = []
        params_feat_else = []
        for name, param in self.feat_net.named_parameters():
            if "cond_fn" in name:
                params_feat_decay.append(param)
            else:
                params_feat_else.append(param)

        opt_params += [
            {
                "params": params_feat_decay,
                "lr": lrs["feat_net"],
                "weight_decay": 1e-4,
            },
            {
                "params": params_feat_else,
                "lr": lrs["feat_net"],
                "weight_decay": wds["feat_net"],
            },
            {
                "params": self.energy_net.parameters(),
                "lr": lrs["energy_net"],
                "weight_decay": wds["energy_net"],
            },
        ]

        return opt_params

    def precompute_pc_feats(self, pc_surf):
        pc_surf = self.tform_pc_to_sdf_domain(pc_surf)

        # pc features and latent
        plane_feat = self.pointnet.get_plane_features(pc_surf)
        # 3 x [B, 256, 64, 64]
        ori_feat = torch.cat(plane_feat, dim=1)
        vae_out = self.vae_model(ori_feat)
        recon_plane_feat, latent = vae_out[0], vae_out[-1]
        # recon: [B, 768, 64, 64]
        # latent: [B, 768]

        self.recon_plane_feat = recon_plane_feat

    def unset_precompute(self):
        self.recon_plane_feat = None

    def set_dom_params(self, params):
        self.dom_params = params

    def unset_dom_params(self):
        self.dom_params = None

    def tform_pc_to_sdf_domain(self, pc):
        """
        pc: B x N(#points) x 3
        assumes pc is at data scale (8.0)
        """
        mesh_load_scale = (
            self.dom_params["mesh_load_scale"].unsqueeze(1).unsqueeze(1).float()
        )
        mesh_center = self.dom_params["norm_center"].unsqueeze(1).float()
        mesh_scale = self.dom_params["norm_scale"].unsqueeze(1).unsqueeze(1).float()
        data_scale = self.dom_params["data_scale"].unsqueeze(1).unsqueeze(1).float()
        pc_surf_mean = self.dom_params["pc_surf_mean"].unsqueeze(1).float()

        rep = pc.shape[1]
        mesh_load_scale = mesh_load_scale.repeat_interleave(rep, dim=1)
        mesh_center = mesh_center.repeat_interleave(rep, dim=1)
        mesh_scale = mesh_scale.repeat_interleave(rep, dim=1)
        data_scale = data_scale.repeat_interleave(rep, dim=1)
        pc_surf_mean = pc_surf_mean.repeat_interleave(rep, dim=1)

        # undo recenter/rescale
        pc = (pc + pc_surf_mean) / data_scale

        # pc = pc / mesh_load_scale
        # # mesh to sdf domain (normalized)
        # pc = pc - mesh_center
        # pc = pc * mesh_scale
        # Below is equivalent to above operations
        pc = pc * (mesh_scale / mesh_load_scale) - mesh_scale * mesh_center
        return pc

    def forward(self, H, k):
        # 0. represent H with points
        p = map_projected_points(H, self.points_gripper)  # [B*N, 30 , 3]
        k_ext = k.unsqueeze(1).repeat(1, p.shape[1])  # [B*N, npts]

        # 1 get object encoding
        B = self.recon_plane_feat.shape[0]
        N = H.shape[0] // B
        npts = p.shape[1]
        p = p.reshape(B, N * npts, 3)
        with torch.no_grad():
            # convert points to sdf domain to query features
            p_sdf = self.tform_pc_to_sdf_domain(p)
            pt_feats = self.pointnet.forward_with_plane_features(
                self.recon_plane_feat, p_sdf
            )  # [B, N*npts, 256]

        z_ext = pt_feats.reshape(B * N, npts, -1)
        p = p.reshape(B * N, npts, 3)

        # 2. compute features
        psi = self.feat_net(p, k_ext, z_ext)

        # 3. compute energy
        psi_flatten = psi.reshape(psi.shape[0], -1)
        e = self.energy_net(psi_flatten)
        return e

    def compute_sdf(self, pts):
        # expecting pts already be in data domain

        # get object encoding (this assumes in sdf domain)
        pt_feats = self.pointnet.forward_with_plane_features(
            self.recon_plane_feat, pts
        )  # [B, npts, 256]

        k = torch.rand_like(pts[..., 0])
        feats = self.feat_net(pts, k, pt_feats)
        return feats[..., 0]

    def train_step(self, data_batch):
        pc_surf = data_batch["pc_surf"]
        H_grasps = data_batch["H_grasps"]
        sdf_xyz = data_batch["sdf_xyz"]
        sdf_gt = data_batch["sdf_dist"]
        B, N = H_grasps.shape[:2]
        loss = 0.0

        dom_params = {
            "pc_surf_mean": data_batch["pc_surf_mean"],
            "mesh_load_scale": data_batch["mesh_load_scale"].float(),
            "norm_center": data_batch["norm_center"],
            "norm_scale": data_batch["norm_scale"].float(),
            "data_scale": data_batch["data_scale"].float(),
        }
        self.set_dom_params(dom_params)

        # pre-compute
        self.precompute_pc_feats(pc_surf=pc_surf)

        # denoising loss
        H_grasps = H_grasps.view(-1, 4, 4)
        grad_energy, z_target = losses.pre_loss_projected_denoising_gdiff(
            H=H_grasps, model=self, sigma=0.5, eps=1e-5
        )
        loss_denoising = losses.loss_projected_denoising_gdiff(grad_energy, z_target)
        loss_denoise = loss_denoising.mean()

        sdf_pred = self.compute_sdf(sdf_xyz)
        sdf_gt = sdf_gt.clamp(min=-1.0, max=1.0)  # since tanh

        loss_sdf = losses.loss_sdf_gdiff(sdf_pred, sdf_gt)

        # final loss (for backprop)
        loss += self.loss_cfg["denoise_loss_weight"] * loss_denoise
        loss += self.loss_cfg["sdf_loss_weight"] * loss_sdf

        metrics_dict = {
            "loss": loss.detach().cpu(),
            "loss_denoise": loss_denoise.detach().cpu(),
            "loss_sdf": loss_sdf.detach().cpu(),
        }

        # reset
        self.unset_precompute()
        self.unset_dom_params()
        return loss, metrics_dict

    def sample_grasps(self, data_sample: Dict, num_samples: int = 10) -> torch.Tensor:
        """
        Defined for a single sample (converted into batch size: 1)
        """
        # add batch dimension
        pc_surf = data_sample["pc_surf"]
        pc_surf = pc_surf.unsqueeze(0)

        dom_params = {
            "pc_surf_mean": data_sample["pc_surf_mean"].unsqueeze(0),
            "mesh_load_scale": data_sample["mesh_load_scale"].unsqueeze(0),
            "norm_center": data_sample["norm_center"].unsqueeze(0),
            "norm_scale": data_sample["norm_scale"].unsqueeze(0),
            "data_scale": data_sample["data_scale"].unsqueeze(0).float(),
        }
        self.set_dom_params(dom_params)

        # pre-compute pc conditioning
        self.precompute_pc_feats(pc_surf=pc_surf)

        # langevin dynamics
        lang_dyn = AnnLangDynSE3(
            model=self,
            T=70,
            k_steps=2,
            T_fit=50,
            sdf_cost=False,
            sdf_grad_align=False,
            device="cuda:0",
        )
        H = lang_dyn.sample(batch_size=num_samples)

        # reset
        self.unset_precompute()
        self.unset_dom_params()
        return H


def get_meta_module(arch, net_cfg, opt_cfg):
    # main method to fetch meta modules for training/testing
    if arch == "VSIGD":
        return MetaVSIGD(net_cfg, opt_cfg)
    else:
        raise ValueError(f"Unknown arch: {arch}")
