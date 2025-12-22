"""
Infer grasps on arbitrary point clouds
"""

from infer import *


class InferRw(Infer):
    def __init__(self, cfg_dict: Dict):
        # config
        model_cfg = cfg_dict["model"]
        self.ckpt_dir = Path(model_cfg["ckpt_dir"])
        self.device = torch.device("cuda:0")

        opt_cfg = cfg_dict["optimizer"]

        # model
        net_cfg = model_cfg["NetworkSpecs"]
        arch = net_cfg.get("arch", None)
        print(f"Model architecture: {arch}")

        self.model = get_meta_module(arch, net_cfg, opt_cfg)
        self.model = self.model.to(self.device)
        self.model.eval()

    def infer_on_pointcloud(
        self, pc_surf: np.ndarray, ngrasps: int = 15, vis: bool = False
    ):
        pc_surf = torch.from_numpy(pc_surf)

        data_scale = 8.0  # scale to nn space

        # sdf params
        pc_max, _ = torch.max(pc_surf, dim=0)
        pc_min, _ = torch.min(pc_surf, dim=0)
        norm_center = (pc_max + pc_min) / 2.0
        pc_copy = pc_surf.detach().clone()
        pc_copy -= norm_center  # center the point cloud
        max_dist = torch.sqrt(torch.max(torch.sum(pc_copy**2, dim=-1)))
        norm_scale = 1.0 / max_dist

        # nn space
        pc_surf = pc_surf * data_scale  # scale to nn space
        pc_mean = pc_surf.mean(axis=0)
        pc_surf -= pc_mean  # center the point cloud

        data_sample = {
            "pc_surf": pc_surf,
            "pc_surf_mean": pc_mean,
            "mesh_load_scale": 1.0,
            "data_scale": data_scale,
            "norm_scale": norm_scale,
            "norm_center": norm_center,
        }

        data_sample = torch_utils.tensorsDict2Device(data_sample, device=self.device)

        # ----- sample grasps -----
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        H_grasps_pred = self.sample_grasps(
            data_sample=data_sample,
            num_samples=ngrasps,
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tdelta = t1 - t0
        print(" - Sampling time:", tdelta)
        # -------------------------

        # convert to original frame
        H_grasps_pred, _, pc_surf = self.cvt_pred_to_vis_frame(
            data_sample, H_grasps_pred
        )

        print(f"Predicted grasps: {H_grasps_pred.shape}")
        H_grasps_pred = H_grasps_pred.detach().cpu().numpy()

        if vis:
            scene_geoms = grasp_vis.get_scene_geoms_o3d(
                mesh_obj=None,
                pc_surf=pc_surf.cpu().numpy(),
                pc_surf_mean=pc_mean.cpu().numpy(),
                rand_H=None,
                H_grasps_pred=H_grasps_pred,
                dataset_scale=data_scale,
                gripper_scale=1.0,
            )
            scene_geoms = [list(dict_obj.values())[0] for dict_obj in scene_geoms]
            o3d.visualization.draw_geometries(scene_geoms)

        return H_grasps_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-dir", type=Path, required=True)
    parser.add_argument("--ckpt-filename", type=str, required=True)
    parser.add_argument("--object-pc", type=Path, required=True)
    parser.add_argument(
        "--ngrasps", type=int, default=15, help="Number of grasps to sample"
    )
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()

    assert (
        args.object_pc.exists()
    ), f"Object point cloud file not found: {args.object_pc}"

    session_name = args.cfg_dir.name
    print(f"> Loading session: {session_name}")

    assert args.cfg_dir.exists(), f"Session directory not found: {args.cfg_dir}"
    cfg_file = list(args.cfg_dir.glob("*.yml"))
    assert len(cfg_file) == 1, f"Found {len(cfg_file)} config files in {args.cfg_dir}"
    cfg_file = cfg_file[0]

    cfg_dict = cmn.parse_yaml_file(filepath=cfg_file)
    # log configurations
    print("*" * 20)
    print("Configurations:")
    print(cfg_dict)
    print("*" * 20)

    vis_cls = InferRw(cfg_dict=cfg_dict)
    vis_cls.load_ckpt(session_name=session_name, ckpt_filename=args.ckpt_filename)

    object_pc = np.load(args.object_pc).astype(np.float32)
    H_grasps_pred = vis_cls.infer_on_pointcloud(
        pc_surf=object_pc, ngrasps=args.ngrasps, vis=args.vis
    )

    # write grasps
    gwrite_fp = args.object_pc.with_suffix(".grasps.npy")
    np.save(gwrite_fp, H_grasps_pred)
    print(f"Grasps saved to: {gwrite_fp}")
