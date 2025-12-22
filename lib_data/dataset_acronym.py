import os
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import trimesh
import open3d as o3d
import h5py
from torch.utils.data import Dataset
from scipy.stats import special_ortho_group

from lib_data.partial_pc import (
    render_partial_pointcloud,
    views_to_tfs,
    get_pc_mask_per_H,
)


def get_fibonacci_sphere(num_points):
    points = []

    phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points += [np.array([x, y, z])]

    points = np.stack(points)

    return points


class AcronymGrasps:
    def __init__(self, filename, data_dir=None):
        self.data_dir = data_dir

        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode("utf-8")
            self.mesh_type = self.mesh_fname.split("/")[1]
            self.mesh_id = self.mesh_fname.split("/")[-1].split(".")[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            self.mesh_fname = data["object/file"][()].decode("utf-8")
            self.mesh_type = self.mesh_fname.split("/")[1]
            self.mesh_id = self.mesh_fname.split("/")[-1].split(".")[0]
            self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)

        self.grasps, self.success = self.load_grasps(filename)
        good_idxs = np.argwhere(self.success == 1)[:, 0]
        # bad_idxs = np.argwhere(self.success == 0)[:, 0]
        self.good_grasps = self.grasps[good_idxs, ...]
        # self.bad_grasps = self.grasps[bad_idxs, ...]

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            T = np.array(data["grasps/transforms"])
            success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success

    def load_mesh(self):
        mesh_path_file = os.path.join(self.data_dir, self.mesh_fname)
        mesh = trimesh.load(mesh_path_file, file_type="obj", force="mesh")

        mesh.apply_scale(self.mesh_scale)
        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh

    def load_mesh_o3d(self):
        mesh_path_file = os.path.join(self.data_dir, self.mesh_fname)
        mesh = o3d.io.read_triangle_mesh(mesh_path_file)
        mesh.scale(self.mesh_scale, center=[0, 0, 0])
        return mesh

    def sdf_filename_from_mesh_fname(self):
        mesh_fname = self.mesh_fname
        mesh_type = mesh_fname.split("/")[1]
        mesh_name = mesh_fname.split("/")[-1]
        filename = mesh_name.split(".obj")[0]
        sdf_filepath = Path(self.data_dir) / "sdf" / mesh_type / (filename + ".json")
        return sdf_filepath


def get_grasp_files(data_dir, class_type, filter_good_grasps=True):
    grasps_dir = os.path.join(data_dir, "grasps")
    grasp_files = []
    for class_type_i in class_type:
        cls_grasps_files = sorted(glob.glob(grasps_dir + "/" + class_type_i + "/*.h5"))

        for grasp_file in cls_grasps_files:
            g_obj = AcronymGrasps(grasp_file, data_dir=data_dir)

            ## Grasp File ##
            if filter_good_grasps and g_obj.good_grasps.shape[0] > 0:
                grasp_files.append(grasp_file)
            elif not filter_good_grasps:
                grasp_files.append(grasp_file)
    return grasp_files


class PointcloudAcronymAndNglodSDFDataset(Dataset):
    """
    for v3 dataset, where we explicitly sample sdf points near surface vs random
    TODO: add points away from the (-1, +1) block.
    """

    def __init__(
        self,
        data_dir: Path,
        class_type=("Mug",),
        n_pointcloud=1000,
        n_density=200,
        n_coords=1000,
        use_sdf=True,
        augmented_rotation=True,
        one_obj=False,
        partial=False,
        sdf_data_path="",
        signed_dist=True,
    ):
        self.class_type = class_type
        self.data_dir = data_dir
        self.sdf_data_path = self.data_dir / sdf_data_path
        assert self.sdf_data_path.exists(), f"Path does not exist: {self.sdf_data_path}"

        # TODO: use a list of object instances for training/validation
        self.grasp_files = get_grasp_files(self.data_dir, class_type)

        ## Variables on Data
        self.n_pointcloud = n_pointcloud
        self.n_density = n_density
        self.n_occ = n_coords
        self.use_sdf = use_sdf
        self.augmented_rotation = augmented_rotation
        self.one_obj = one_obj
        self.partial = partial
        self.signed_dist = signed_dist

        ## Visualization
        self.scale = 8.0

    def __len__(self):
        if self.one_obj:
            return 1
        return len(self.grasp_files)

    # ----------- helpers -----------
    def get_graspobj_for_object(self, obj_idx):
        grasps_obj = AcronymGrasps(self.grasp_files[obj_idx], data_dir=self.data_dir)
        return grasps_obj

    def _get_grasps(self, grasp_obj, pc=None):
        """
        pc: use point cloud to select corresponding gt grasps
        """
        H_all = grasp_obj.good_grasps
        if pc is not None:  # for training with partial point clouds
            pts_thres = 10
            H_grasps = np.zeros((0, 4, 4), dtype=np.float32)
            rix = np.arange(H_all.shape[0])
            np.random.shuffle(rix)
            s = 0
            w = 10
            c = 0
            c_max = 5
            while True:
                idxs = rix[s : s + self.n_density * w]
                H_samp = H_all[idxs]

                # check for validity
                H_samp_msk = get_pc_mask_per_H(H_samp, pc)
                H_samp_msk = H_samp_msk.sum(axis=-1) >= pts_thres
                H_samp = H_samp[H_samp_msk]

                s += self.n_density * w
                if H_samp.shape[0] == 0:
                    c += 1
                    if c >= c_max:
                        # avoiding infinite loops
                        return None
                    continue
                H_grasps = np.concatenate([H_grasps, H_samp], axis=0)
                if H_grasps.shape[0] >= self.n_density:
                    H_grasps = H_grasps[: self.n_density]
                    break
                if s >= len(rix):
                    s = 0
                    c += 1
                    if c >= c_max:
                        # avoiding infinite loops
                        return None
                    continue
        else:
            rix = np.random.randint(low=0, high=H_all.shape[0], size=self.n_density)
            H_grasps = H_all[rix, ...]
        return H_grasps

    def _get_mesh_pc_partial(self, grasp_obj):
        mesh = grasp_obj.load_mesh()
        # sample pcs from centered mesh
        center = mesh.centroid
        mesh = mesh.apply_translation(-center)

        width, height = 1000, 1000
        fx = fy = 200.0  # focal length in pixels (tweak as needed)
        near, far = 0.001, 200
        num_views = 1000
        view_vecs = get_fibonacci_sphere(num_views)
        # sample only 1 view:
        while True:
            rand_idx = np.random.randint(0, num_views)
            view_vecs_sub = view_vecs[[rand_idx]]
            camTs = views_to_tfs(mesh, view_vecs_sub)
            camT = camTs[0]

            # trimesh + embree
            pc, *_ = render_partial_pointcloud(
                mesh,
                camT,
                fx,
                fy,
                width,
                height,
                near=near,
                far=far,
                n_points=self.n_pointcloud,
            )
            pc = pc + center
            if pc.shape[0] == self.n_pointcloud:
                break
        assert pc.shape == (self.n_pointcloud, 3)
        return pc

    def _get_pc_sdf_data(self, grasp_obj):
        """
        SDF data (normalized)
        """
        # data dist: 70% near-surface, 30% random
        near_surf_count = int(self.n_occ * 0.7)
        rand_count = self.n_occ - near_surf_count

        # obj_fp_base = self.gt_files[idx]
        filename = grasp_obj.sdf_filename_from_mesh_fname()
        cat = filename.parent.stem
        obj = filename.stem
        obj_fp_base = str(self.sdf_data_path / cat / f"{obj}")

        if not self.partial:
            surf_pts = np.load(obj_fp_base + "-trace-pts.npy", mmap_mode="r")
            assert (
                surf_pts.shape[0] == 100000
            ), f"{obj_fp_base} surf_pts shape: {surf_pts.shape}"
        # surf_sdf = np.load(obj_fp_base + "-trace-sdf.npy", mmap_mode="r")
        near_pts = np.load(obj_fp_base + "-near-pts.npy", mmap_mode="r")
        near_sdf = np.load(obj_fp_base + "-near-sdf.npy", mmap_mode="r")
        rand_pts = np.load(obj_fp_base + "-rand-pts.npy", mmap_mode="r")
        rand_sdf = np.load(obj_fp_base + "-rand-sdf.npy", mmap_mode="r")
        center = np.load(obj_fp_base + "-v_center.npy")
        scale = np.load(obj_fp_base + "-v_scale.npy")

        assert near_pts.shape[0] == math.ceil(
            600000 * 0.7
        ), f"{obj_fp_base} near_pts shape: {near_pts.shape}"
        assert rand_pts.shape[0] == math.ceil(
            600000 * 0.3
        ), f"{obj_fp_base} rand_pts shape: {rand_pts.shape}"

        def _idxs_from_count(npts, nsamples):
            if npts < nsamples:
                idxs = np.random.choice(range(npts), size=(nsamples,), replace=True)
            else:
                idxs = np.random.permutation(range(npts))[:nsamples]
            return idxs

        # sample near-surf pts
        half = int(near_surf_count / 2)
        near_pos_mask = (near_sdf > 0)[:, 0]
        near_neg_mask = (near_sdf < 0)[:, 0]
        near_pos_idxs = _idxs_from_count(near_pos_mask[near_pos_mask].shape[0], half)
        if np.sum(near_neg_mask) < 1:
            # sample from positives
            near_neg_idxs = _idxs_from_count(
                near_pos_mask[near_pos_mask].shape[0], half
            )
        else:
            near_neg_idxs = _idxs_from_count(
                near_neg_mask[near_neg_mask].shape[0], half
            )

        near_pts_pos = near_pts[near_pos_mask][near_pos_idxs]
        near_sdf_pos = near_sdf[near_pos_mask][near_pos_idxs]
        if np.sum(near_neg_mask) < 1:
            # sample from positives
            near_pts_neg = near_pts[near_pos_mask][near_neg_idxs]
            near_sdf_neg = near_sdf[near_pos_mask][near_neg_idxs]
        else:
            near_pts_neg = near_pts[near_neg_mask][near_neg_idxs]
            near_sdf_neg = near_sdf[near_neg_mask][near_neg_idxs]

        # sample rand pts
        half = int(rand_count / 2)
        rand_pos_mask = (rand_sdf > 0)[:, 0]
        rand_neg_mask = (rand_sdf < 0)[:, 0]
        rand_pos_idxs = _idxs_from_count(rand_pos_mask[rand_pos_mask].shape[0], half)
        if np.sum(rand_neg_mask) < 1:
            # sample from positives
            rand_neg_idxs = _idxs_from_count(
                rand_pos_mask[rand_pos_mask].shape[0], half
            )
        else:
            rand_neg_idxs = _idxs_from_count(
                rand_neg_mask[rand_neg_mask].shape[0], half
            )

        rand_pts_pos = rand_pts[rand_pos_mask][rand_pos_idxs]
        rand_sdf_pos = rand_sdf[rand_pos_mask][rand_pos_idxs]
        if np.sum(rand_neg_mask) < 1:
            # sample from positives
            rand_pts_neg = rand_pts[rand_pos_mask][rand_neg_idxs]
            rand_sdf_neg = rand_sdf[rand_pos_mask][rand_neg_idxs]
        else:
            rand_pts_neg = rand_pts[rand_neg_mask][rand_neg_idxs]
            rand_sdf_neg = rand_sdf[rand_neg_mask][rand_neg_idxs]

        if not self.partial:
            # sample surf pts
            surf_pts_idxs = np.random.permutation(range(surf_pts.shape[0]))[
                : self.n_pointcloud
            ]
            surf_pts = surf_pts[surf_pts_idxs]
        else:
            # sample pcs later
            surf_pts = None

        if not self.signed_dist:
            # for unsigned distance
            near_sdf_neg = -near_sdf_neg
            rand_sdf_neg = -rand_sdf_neg

        pts_xyz = np.concatenate(
            [
                near_pts_pos,
                near_pts_neg,
                rand_pts_pos,
                rand_pts_neg,
            ],
            axis=0,
        )
        pts_sdf = np.concatenate(
            [
                near_sdf_pos,
                near_sdf_neg,
                rand_sdf_pos,
                rand_sdf_neg,
            ],
            axis=0,
        )
        return surf_pts, pts_xyz, pts_sdf, center, scale

    # ----------- loader -----------
    def __getitem__(self, index):
        if self.one_obj:
            index = 0

        done = False
        while not done:
            ## Fetch grasp obj ##
            grasps_obj = self.get_graspobj_for_object(index)

            ## get surface pc, sdf-points, and mesh params
            assert self.use_sdf, "SDF must be used for this dataset"
            pc_surf, xyz, sdf, center, scale = self._get_pc_sdf_data(grasps_obj)
            mesh_load_scale = grasps_obj.mesh_scale

            if self.partial:
                assert pc_surf is None
                H_grasps = None
                for _ in range(5):  # tries
                    pc_surf = self._get_mesh_pc_partial(grasps_obj)
                    ## Sample grasps (good only)
                    H_grasps = self._get_grasps(grasps_obj, pc=pc_surf)
                    if H_grasps is not None:
                        done = True
                        break
                if not done:
                    index = (
                        (random.randint(0, len(self) - 1)) if not self.one_obj else 0
                    )
            else:
                # sdf-domain to workspace
                pc_surf = pc_surf * (mesh_load_scale / scale) + mesh_load_scale * center
                ## Sample grasps (good only)
                H_grasps = self._get_grasps(grasps_obj)
                done = True

        xyz = xyz * (mesh_load_scale / scale) + mesh_load_scale * center
        sdf = sdf * (mesh_load_scale / scale)
        sdf = np.squeeze(sdf, axis=-1)

        # ------- rescale
        pc_surf *= self.scale
        H_grasps[..., :3, -1] *= self.scale
        xyz = xyz * self.scale
        sdf = sdf * self.scale

        # ------- recenter
        pc_surf_mean = np.mean(pc_surf, axis=0)
        pc_surf -= pc_surf_mean
        H_grasps[..., :3, -1] -= pc_surf_mean
        xyz -= pc_surf_mean

        if self.augmented_rotation:
            # sample random rotation
            rand_R = special_ortho_group.rvs(3)
            rand_H = np.eye(4)
            rand_H[:3, :3] = rand_R

            # apply random rotation
            pc_surf = np.einsum("mn,bn->bm", rand_R, pc_surf)
            H_grasps = np.einsum("mn,bnk->bmk", rand_H, H_grasps)
            xyz = np.einsum("mn,bn->bm", rand_R, xyz)

        data_sample = {
            "index": torch.tensor([index]).long(),
            "H_grasps": torch.from_numpy(H_grasps).float(),  # 6DoF grasps
            "pc_surf": torch.from_numpy(pc_surf).float(),  # surface point cloud
            "pc_surf_mean": torch.from_numpy(
                pc_surf_mean
            ).float(),  # mean of surface point cloud
            "mesh_load_scale": mesh_load_scale,
            "norm_scale": scale,
            "norm_center": torch.from_numpy(center).float(),
            "data_scale": self.scale,
            # for sdf
            "sdf_xyz": torch.from_numpy(xyz).float(),  # sample sdf locations
            "sdf_dist": torch.from_numpy(sdf).float(),  # sdf values at xyz
        }
        if self.augmented_rotation:
            data_sample["rand_R"] = torch.from_numpy(rand_R).float()
        return data_sample
