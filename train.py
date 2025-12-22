import argparse
import copy
import shutil
import datetime
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils import tensorboard as tb
from rich import print
from rich.progress import track as rich_trackbar

from lib_nn.meta_modules import get_meta_module
from lib_nn import torch_utils
from lib_nn import metrics
from lib_data import dataset_acronym
from lib_data import grasp_vis
from lib_data import common as cmn


class Train:
    def __init__(self, cfg_dict: Dict, args: argparse.Namespace):
        # config
        data_cfg = cfg_dict["data"]
        data_dir = Path(data_cfg["data_dir"])
        class_type = data_cfg["class_type"]
        self.render_idxs = data_cfg.get("render_idxs", None)
        grasp_density = data_cfg["grasp_density"]
        pc_points = data_cfg["pc_points"]
        sdf_points = data_cfg["sdf_points"]
        signed_dist = data_cfg.get("signed_dist", True)

        model_cfg = cfg_dict["model"]
        ckpt_dir = Path(model_cfg["ckpt_dir"])
        self.device = torch.device("cuda:0")

        opt_cfg = cfg_dict["optimizer"]
        self.total_epochs = opt_cfg["epochs"]
        self.epoch_save_period = opt_cfg["epoch_save_period"]
        self.render_save_period = opt_cfg.get("render_save_period", None)
        self.epoch_now = 0

        # datasets
        train_cfg = cfg_dict["train"]
        DatasetClass = dataset_acronym.PointcloudAcronymAndNglodSDFDataset

        ds_args = {
            "data_dir": data_dir,
            "sdf_data_path": data_cfg["sdf_data_subdir"],
            "class_type": class_type,
            "n_pointcloud": pc_points,
            "n_density": grasp_density,
            "n_coords": sdf_points,
            "augmented_rotation": True,
            "one_obj": args.one_obj,
            "signed_dist": signed_dist,
            "partial": data_cfg.get("partial_pc", False),
        }

        self.train_set = DatasetClass(**ds_args)
        train_dl_params = {
            "batch_size": train_cfg["batch_size"],
            "shuffle": True,
            "drop_last": False,
            "num_workers": train_cfg["num_workers"],
            "pin_memory": True,
        }
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, **train_dl_params
        )

        if self.render_idxs is not None:
            ds_args_val = copy.deepcopy(ds_args)
            ds_args_val["augmented_rotation"] = False  # no aug for val
            self.val_set = DatasetClass(**ds_args_val)

        # model
        folder_name = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        if args.desc:
            folder_name = f"{folder_name}_{args.desc}"
        self.sess_dir = ckpt_dir / folder_name
        self.sess_dir.mkdir(exist_ok=False)
        self.ckpt_save_dir = self.sess_dir / "checkpoints"
        self.ckpt_save_dir.mkdir()
        # rendering grasps
        if self.render_idxs is not None:
            if args.one_obj:
                self.render_idxs = [0]
            self.render_dir = self.sess_dir / "renders"
            self.render_dir.mkdir(exist_ok=False)

        net_cfg = model_cfg["NetworkSpecs"]
        arch = net_cfg.get("arch", None)
        self.model = get_meta_module(arch, net_cfg, opt_cfg)
        self.model = self.model.to(self.device)

        # opt
        opt_params = self.model.get_opt_params()
        self.opt = torch.optim.Adam(opt_params, amsgrad=opt_cfg.get("amsgrad", False))

        # tensorboard
        self.writer = tb.SummaryWriter(log_dir=str(self.sess_dir / "tensorboard"))

        # backup cfg file
        shutil.copy2(args.cfg, self.sess_dir)

        # metrics
        self.best_train_loss = torch.inf

    def train_step(self, data_batch) -> Tuple[torch.tensor]:
        self.model.train()
        self.opt.zero_grad()

        data_batch = torch_utils.tensorsDict2Device(data_batch, device=self.device)

        loss, metrics_dict = self.model.train_step(data_batch)
        loss.backward()
        self.opt.step()

        return metrics_dict

    def do_epoch(self, epoch: int):
        torch.cuda.reset_peak_memory_stats(device=self.device)
        epoch_metrics_dict = dict()

        # iterations
        for data_batch in rich_trackbar(
            self.train_loader, description=f"Working on train set"
        ):
            step_metrics_dict = self.train_step(data_batch=data_batch)

            for met in step_metrics_dict:
                if met not in epoch_metrics_dict:
                    epoch_metrics_dict[met] = torch.zeros(1, dtype=torch.float32).cpu()
                epoch_metrics_dict[met] += step_metrics_dict[met]

        per_epoch_dict = self.per_epoch_metrics(
            metrics_dict=epoch_metrics_dict,
            iterations=len(self.train_loader),
            train_or_val="train",
        )
        mem_max_B = torch.cuda.max_memory_allocated(device=self.device)
        per_epoch_dict["mem_max_GB"] = mem_max_B / (1024**3)

        self.log_scalars_to_tensorboard(scalar_dict=per_epoch_dict, epoch=epoch)
        return per_epoch_dict

    def sample_grasps(self, *args, **kwargs):
        return self.model.sample_grasps(*args, **kwargs)

    @staticmethod
    def cvt_pred_to_vis_frame(data_sample, H_grasps_pred):
        # ------------------------
        # convert to original frame
        pc_surf = data_sample["pc_surf"]
        pc_surf_mean = data_sample["pc_surf_mean"]
        H_grasps_gt = data_sample.get("H_grasps", None)
        data_scale = data_sample["data_scale"]

        # grasps (pred)
        H_grasps_pred = H_grasps_pred.clone()
        H_grasps_pred[..., :3, -1] = H_grasps_pred[..., :3, -1] + pc_surf_mean
        H_grasps_pred[..., :3, -1] = H_grasps_pred[..., :3, -1] / data_scale

        # grasps (gt)
        if H_grasps_gt is not None:
            H_grasps_gt = H_grasps_gt.clone()
            H_grasps_gt[..., :3, -1] = H_grasps_gt[..., :3, -1] + pc_surf_mean
            H_grasps_gt[..., :3, -1] = H_grasps_gt[..., :3, -1] / data_scale

        # pc
        pc_surf = pc_surf + pc_surf_mean
        pc_surf = pc_surf / data_scale
        return H_grasps_pred, H_grasps_gt, pc_surf

    def do_eval(self, epoch: int, data_idxs: list):
        dataset = self.val_set

        epoch_metrics_dict = {
            "emd_mu": torch.zeros(1, dtype=torch.float32).cpu(),
            "emd_std": torch.zeros(1, dtype=torch.float32).cpu(),
        }

        num_samples = 200
        for idx in rich_trackbar(data_idxs, description=f"Rendering grasps"):
            # fetch data
            data_sample = dataset[idx]
            data_sample = torch_utils.tensorsDict2Device(
                data_sample, device=self.device
            )

            pc_surf = data_sample["pc_surf"]
            H_grasps_gt = data_sample["H_grasps"]

            # inference
            with torch.random.fork_rng(devices=[self.device]):
                torch.random.manual_seed(123)

                H_grasps_pred = self.sample_grasps(
                    data_sample=data_sample,
                    num_samples=num_samples,
                )

            # ------------------------
            # convert to original frame
            H_grasps_pred, H_grasps_gt, pc_surf = self.cvt_pred_to_vis_frame(
                data_sample, H_grasps_pred
            )

            # ------------ checks on grasps
            H_grasps_pred = H_grasps_pred.cpu().numpy()
            assert len(H_grasps_pred) == num_samples
            H_grasps_gt = H_grasps_gt.cpu().numpy()
            H_grasps_gt = H_grasps_gt[:num_samples]

            # compute emd
            emd_mu, emd_std = metrics.empirical_emd(
                H_grasps_gt, H_grasps_pred, n_grasps=num_samples
            )

            epoch_metrics_dict["emd_mu"] += emd_mu
            epoch_metrics_dict["emd_std"] += emd_std**2

            # ---------- vis ----------
            # fetch object
            g_obj = dataset.get_graspobj_for_object(obj_idx=idx)

            # only 5 grasps for vis
            n_grasps = 5
            H_grasps_gt = H_grasps_gt[:n_grasps]
            H_grasps_pred = H_grasps_pred[:n_grasps]

            # random rotation
            rand_H = np.eye(4, dtype=np.float32)
            if "rand_R" in data_sample.keys():
                rand_H[:3, :3] = data_sample["rand_R"].cpu().numpy()

            # open3d
            scene_geoms = grasp_vis.get_scene_geoms_o3d(
                mesh_obj=g_obj.load_mesh_o3d(),
                pc_surf=pc_surf.cpu().numpy(),
                pc_surf_mean=data_sample["pc_surf_mean"].cpu().numpy(),
                rand_H=rand_H,
                H_grasps_gt=H_grasps_gt,
                H_grasps_pred=H_grasps_pred,
                dataset_scale=dataset.scale,
            )

            # metadata for saving
            grasps_obj = dataset.get_graspobj_for_object(idx)
            obj_cat = grasps_obj.mesh_type
            obj_id = grasps_obj.mesh_id
            objcatid = f"{obj_cat}_{obj_id}"

            grasp_vis.render_write_o3d(
                scene_geoms, self.render_dir / f"{epoch:06d}_{idx:04d}_{objcatid}.jpg"
            )

        per_epoch_dict = self.per_epoch_metrics(
            epoch_metrics_dict, len(data_idxs), "val"
        )
        self.log_scalars_to_tensorboard(scalar_dict=per_epoch_dict, epoch=epoch)

    def train_val_loop(self):
        # ----------- some stats ------------
        print("Samples per epoch:")
        print(f"Train: {len(self.train_set)}")
        print("Iterations per epoch:")
        print(f"Train: {len(self.train_loader)}")
        # -----------------------------------

        for epoch in range(self.epoch_now, self.total_epochs):
            print("-" * 20)
            print(" > Epoch {}/{}".format(epoch + 1, self.total_epochs))

            # read and log learning rate:
            self.log_scalars_to_tensorboard(
                {
                    f"lr/param_group_{i}": pg["lr"]
                    for i, pg in enumerate(self.opt.param_groups)
                },
                epoch,
            )

            torch.cuda.synchronize()
            t0 = datetime.datetime.now()

            train_dict = self.do_epoch(epoch=epoch)

            torch.cuda.synchronize()
            tn = datetime.datetime.now()

            print(" Epoch Time: {}".format(tn - t0))

            train_loss = train_dict["loss/train"]
            if (epoch + 1) % self.epoch_save_period == 0:
                self.save_checkpoint(epoch + 1, train_loss)
                if train_loss < self.best_train_loss:
                    self.best_train_loss = train_loss
                    self.save_checkpoint(epoch + 1, train_loss, best=True)

            # render grasps
            if (self.render_save_period > 0) and (self.render_idxs is not None):
                if (epoch + 1) % self.render_save_period == 0:
                    self.do_eval(epoch=epoch + 1, data_idxs=self.render_idxs)

            self.epoch_now += 1

        self.writer.flush()
        self.writer.close()
        print("-" * 20)
        print("Training Finished!")
        print("-" * 20)

    @staticmethod
    def per_epoch_metrics(metrics_dict, iterations, train_or_val):
        """displays and returns per_epoch mean of provided metrics"""
        per_epoch_dict = {}
        for key in metrics_dict:
            per_epoch_dict[key + "/" + train_or_val] = metrics_dict[key].item() / float(
                iterations
            )
        print_list = [
            " {}: {} ".format(key, "%.4f" % per_epoch_dict[key])
            for key in per_epoch_dict
        ]
        print("|".join(print_list))
        return per_epoch_dict

    def save_checkpoint(self, idx: int, train_loss: float, best: bool = False):
        if best:
            save_path = self.ckpt_save_dir / f"ckpt_best.pth"
        else:
            save_path = self.ckpt_save_dir / f"ckpt_{idx:03}_{train_loss}.pth"
        state_dict = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "epoch": idx,
            "train_loss": train_loss,
        }
        torch_utils.torch_save_dict(d=state_dict, filepath=save_path)

    def resume(self, sess_dir: Path):
        ckpt_files = [
            i for i in (sess_dir / "checkpoints").glob("*.pth") if "best" not in i.name
        ]
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.name.split("_")[2]))
        ckpt_file = ckpt_files[-1]
        state_dict = torch.load(ckpt_file)

        self.model.load_state_dict(state_dict["model"])
        try:
            self.opt.load_state_dict(state_dict["opt"])
        except:
            print(f"[red]x Optimizer state couldn't be loaded[/red]")
        self.epoch_now = state_dict["epoch"]
        self.best_train_loss = state_dict["train_loss"]

        print(f"Resuming from checkpoint: {ckpt_file}")
        print(f"Epoch: {self.epoch_now}, Best Train Loss: {self.best_train_loss}")

        # preparation
        shutil.copy2(ckpt_file, self.sess_dir / "checkpoints")
        shutil.rmtree(self.sess_dir / "tensorboard")
        shutil.copytree(sess_dir / "tensorboard", self.sess_dir / "tensorboard")

    def log_scalars_to_tensorboard(self, scalar_dict: Dict, epoch: int):
        """routine to log per-epoch metrics to tensorboard"""
        for key in scalar_dict:
            self.writer.add_scalar(key, scalar_dict[key], epoch)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA device unavailable"
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True, help="Path to config file")
    parser.add_argument(
        "--desc", type=str, required=False, default="", help="Path to log file"
    )
    parser.add_argument(
        "--resume-sess",
        type=Path,
        required=False,
        default=False,
        help="Path to session (to resume)",
    )
    parser.add_argument(
        "--one_obj",
        action="store_true",
        help="Use only 1 object for training (for debugging)",
    )
    args = parser.parse_args()

    cfg_dict = cmn.parse_yaml_file(filepath=args.cfg)
    print("*" * 20)
    print("Configurations:")
    print(cfg_dict)
    print("*" * 20)

    if cfg_dict["data"].get("partial_pc", False):
        mp.set_start_method("spawn", force=True)

    trainer_class = Train(cfg_dict=cfg_dict, args=args)
    if args.resume_sess:
        trainer_class.resume(sess_dir=args.resume_sess)
    trainer_class.train_val_loop()
