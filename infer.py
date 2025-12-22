import time
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import open3d as o3d
from rich import print

from train import Train
from lib_nn.meta_modules import get_meta_module
from lib_nn import torch_utils
from lib_data import dataset_acronym
from lib_data import grasp_vis
from lib_data import common as cmn


class Infer(Train):
    def __init__(self, cfg_dict: Dict):
        # config
        data_cfg = cfg_dict["data"]
        data_dir = Path(data_cfg["data_dir"])
        class_type = data_cfg["class_type"]
        grasp_density = data_cfg["grasp_density"]
        pc_points = data_cfg["pc_points"]
        sdf_points = data_cfg["sdf_points"]

        model_cfg = cfg_dict["model"]
        self.ckpt_dir = Path(model_cfg["ckpt_dir"])
        self.device = torch.device("cuda:0")

        opt_cfg = cfg_dict["optimizer"]

        # datasets
        DatasetClass = dataset_acronym.PointcloudAcronymAndNglodSDFDataset

        self.val_set = DatasetClass(
            data_dir=data_dir,
            sdf_data_path=data_cfg["sdf_data_subdir"],
            class_type=class_type,
            n_pointcloud=pc_points,
            n_density=grasp_density,
            n_coords=sdf_points,
            augmented_rotation=False,
        )

        # model
        net_cfg = model_cfg["NetworkSpecs"]
        arch = net_cfg.get("arch", None)
        print(f"Model architecture: {arch}")

        self.model = get_meta_module(arch, net_cfg, opt_cfg)
        self.model = self.model.to(self.device)

    def load_ckpt(self, session_name, ckpt_filename):
        print(f"> Session: {session_name}")
        print(f"- Loading checkpoint: {ckpt_filename}")

        # load model
        state_dict = torch.load(
            self.ckpt_dir / session_name / "checkpoints" / ckpt_filename
        )

        self.model.load_state_dict(state_dict["model"])  # for point-vae

        print(
            f"+ Loaded @ epoch {state_dict['epoch']}"
            f" (loss: {state_dict['train_loss']})"
        )
        self.session_name = session_name
        self.ckpt_filename = ckpt_filename

    def visualize_on_dataset(
        self,
        num_samples=200,
        render=False,
        save_render=False,
    ):
        dataset = self.val_set

        print(f"Full dataset size: {len(dataset)}")

        save_path = Path(f"tests-dir/{self.session_name}/{self.ckpt_filename}")
        save_path.mkdir(parents=True, exist_ok=True)

        sampling_times = []
        for idx in range(0, len(dataset)):
            print("Processing idx:", idx)
            # fetch data
            data_sample = dataset[idx]
            data_sample = torch_utils.tensorsDict2Device(
                data_sample, device=self.device
            )

            pc_surf = data_sample["pc_surf"]
            H_grasps_gt = data_sample["H_grasps"]

            # inference
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            H_grasps_pred = self.sample_grasps(
                data_sample=data_sample,
                num_samples=num_samples,
            )

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            tdelta = t1 - t0
            print(" - Sampling time:", tdelta)
            sampling_times.append(tdelta)

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

            # metadata for saving
            grasps_obj = dataset.get_graspobj_for_object(idx)
            obj_cat = grasps_obj.mesh_type
            obj_id = grasps_obj.mesh_id
            objcatid = f"{obj_cat}_{obj_id}"

            # write to disk
            # Note: if augmented rotations, undo the random rotation
            # rescaling for sim
            H_grasps_4sim = H_grasps_gt.copy()
            H_grasps_pred_4sim = H_grasps_pred.copy()

            mesh_scale = dataset.get_graspobj_for_object(idx).mesh_scale
            np.save(
                save_path / f"{objcatid}-s{mesh_scale}-grasp-pred.npy",
                H_grasps_pred_4sim,
            )
            np.save(save_path / f"{objcatid}-s{mesh_scale}-grasp-gt.npy", H_grasps_4sim)

            if not render:
                continue

            # ---------- vis ----------
            # fetch object
            g_obj = dataset.get_graspobj_for_object(obj_idx=idx)

            # only 5 grasps for vis
            n_grasps = 5
            H_grasps_gt = H_grasps_gt[:n_grasps]
            H_grasps_pred = H_grasps_pred[:n_grasps]

            # random rotation
            rand_H = np.eye(4, dtype=np.float32)
            if "rand_H" in data_sample.keys():
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

            if save_render:
                grasp_vis.render_write_o3d(
                    scene_geoms, save_path / f"render-{objcatid}.jpg"
                )
            else:
                # live vis
                geoms = [v for g in scene_geoms for v in g.values()]
                o3d.visualization.draw_geometries(geoms)

        sampling_time = f"{np.mean(sampling_times)} +- {np.std(sampling_times)}"
        print(f">> Sampling time: {sampling_time}")

        # write yaml
        sess_metadata = {
            "session_name": self.session_name,
            "ckpt_filename": self.ckpt_filename,
            "sampling_time": sampling_time,
        }
        cmn.write_yaml_file(data=sess_metadata, filepath=save_path / "test-info.yaml")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-dir", type=Path, required=True)
    parser.add_argument("--ckpt-filename", type=str, required=True)
    parser.add_argument("--object-classes", type=str, nargs="+", default=None)
    args = parser.parse_args()

    session_name = args.cfg_dir.name
    print(f"> Loading session: {session_name}")

    assert args.cfg_dir.exists(), f"Session directory not found: {args.cfg_dir}"
    cfg_file = list(args.cfg_dir.glob("*.yml"))
    assert len(cfg_file) == 1, f"Found {len(cfg_file)} config files in {args.cfg_dir}"
    cfg_file = cfg_file[0]

    cfg_dict = cmn.parse_yaml_file(filepath=cfg_file)
    if args.object_classes is not None:
        cfg_dict["data"]["class_type"] = list(args.object_classes)
    # log configurations
    print("*" * 20)
    print("Configurations:")
    print(cfg_dict)
    print("*" * 20)

    vis_cls = Infer(cfg_dict=cfg_dict)
    vis_cls.load_ckpt(session_name=session_name, ckpt_filename=args.ckpt_filename)
    vis_cls.visualize_on_dataset(
        render=True,
        save_render=True,
    )
