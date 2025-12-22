"""
Partial point cloud renderer via raycasting in trimesh.
Headless + multiprocessing friendly. Returns Nx3 point cloud (world coords).
Dependencies: numpy, trimesh. Optional: pyembree for speed.
"""

import numpy as np
import trimesh

from lib_data import grasp_vis

# Try to enable pyembree for much faster ray intersections
try:
    import pyembree  # noqa: F401
except ImportError:
    pass


def views_to_tfs(mesh, view_vecs):
    distance = np.linalg.norm(mesh.extents)
    # set camera poses
    view_unit_vecs = view_vecs / np.linalg.norm(view_vecs, axis=1, keepdims=True)
    cam_z_s = -view_unit_vecs

    while True:
        cam_x_s = -1 + 2 * np.random.rand(len(cam_z_s), 3)
        cam_x_s = cam_x_s - np.sum(cam_x_s * cam_z_s, axis=1, keepdims=True) * cam_z_s

        if np.linalg.norm(cam_x_s, axis=1).any() == 0:
            continue
        else:
            cam_x_s /= np.linalg.norm(cam_x_s, axis=1, keepdims=True)
            break

    cam_y_s = np.cross(cam_z_s, cam_x_s)
    cam_y_s /= np.linalg.norm(cam_y_s, axis=1, keepdims=True)

    cam_Ts = np.tile(np.eye(4), (len(view_vecs), 1, 1))
    cam_Ts[:, :3, :3] = np.stack([cam_x_s, cam_y_s, cam_z_s], axis=2)
    cam_Ts[:, :3, 3] = distance * view_unit_vecs
    return cam_Ts


def build_camera_rays(width, height, fx, fy, cx=None, cy=None, camera_pose=np.eye(4)):
    """
    Build ray origins and directions in world frame for a pinhole camera.
    camera_pose: 4x4 world-from-camera transform (i.e. camera pose in world)
    fx, fy: focal lengths in pixels
    cx, cy: principal point. If None, default to center.
    Returns:
        origins: (H*W, 3)
        directions: (H*W, 3) normalized
    """
    if cx is None:
        cx = (width - 1) / 2.0
    if cy is None:
        cy = (height - 1) / 2.0

    # Pixel grid
    xs = (np.arange(width) - cx) / fx  # normalized image plane x
    ys = (np.arange(height) - cy) / fy  # normalized image plane y
    px, py = np.meshgrid(xs, -ys)  # note: -ys if image y goes down; adjust as needed

    # Camera-space directions (z=1)
    camera_dirs = np.stack([px, py, np.ones_like(px)], axis=-1)  # H x W x 3
    camera_dirs = camera_dirs.reshape(-1, 3)

    # Transform directions to world
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    world_dirs = (R @ camera_dirs.T).T
    norms = np.linalg.norm(world_dirs, axis=1, keepdims=True)
    world_dirs = world_dirs / (norms + 1e-12)

    origins = np.tile(t.reshape(1, 3), (world_dirs.shape[0], 1))
    return origins.astype(np.float64), world_dirs.astype(np.float64)


def render_partial_pointcloud(
    mesh: trimesh.Trimesh,
    camera_pose: np.ndarray,
    fx: float,
    fy: float,
    width: int,
    height: int,
    cx: float = None,
    cy: float = None,
    near: float = 0.01,
    far: float = 10.0,
    n_points: int = None,
):
    """
    Raycast the mesh from the camera_pose and return a partial point cloud.
    - mesh: trimesh.Trimesh (must be in world coordinates)
    - camera_pose: 4x4 world-from-camera transform (camera pose)
    - fx, fy: focal lengths in pixels
    - width/height: image size
    - near/far: clipping planes (only accept hits within)
    - n_points: if not None, randomly subsample to at most n_points
    - use_embree: allow use of pyembree if available
    Returns:
        points: (M, 3) ndarray in world coordinates
        depth_image: (H, W) with depth or np.nan for no hit
        mask: (H, W) boolean hit mask
    """
    origins, directions = build_camera_rays(width, height, fx, fy, cx, cy, camera_pose)

    # ray intersector
    # trimesh has mesh.ray which will use pyembree if installed and available
    # Note: mesh.ray.intersects_id returns face indexes of first hit along ray (or -1)
    #       intersects_location returns hit locations (n, 3) but only for rays that hit
    scene_mesh = mesh  # assume mesh is already a Trimesh

    # intersects_first returns distances along ray to first hit
    # trimesh's API: mesh.ray.intersects_any / intersects_location / intersects_id
    locations, index_ray, _ = scene_mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    # intersects_location returns only hits
    # index_ray gives the ray index per returned hit
    depth = np.full((origins.shape[0],), np.nan, dtype=np.float64)
    if locations.shape[0] > 0:
        # compute signed distance along ray (positive where hit)
        vec = locations - origins[index_ray]
        d = np.linalg.norm(vec, axis=1)
        # set depth at ray positions
        depth[index_ray] = d

    # reshape depth to HxW
    depth_image = depth.reshape((height, width))

    # clip by near/far
    mask = np.isfinite(depth_image) & (depth_image >= near) & (depth_image <= far)

    # compute 3D points: for each pixel index with mask, compute p = o + d * depth
    ys, xs = np.nonzero(mask)
    ray_indices = ys * width + xs
    if ray_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32), depth_image, mask

    selected_origins = origins[ray_indices]
    selected_dirs = directions[ray_indices]
    selected_depths = depth[ray_indices]  # flat depth from earlier

    points = selected_origins + selected_dirs * selected_depths.reshape(-1, 1)

    # optional random subsample
    if n_points is not None and points.shape[0] > n_points:
        idx = np.random.choice(points.shape[0], size=n_points, replace=False)
        points = points[idx]

    return points.astype(np.float32), depth_image, mask


def se3_inv_batched(H):
    """
    Invert batched SE(3) transforms.
    H: N x 4 x 4
    """
    if H.shape[-2:] != (4, 4):
        raise ValueError("Input must have shape (..., 4, 4).")

    R = H[..., :3, :3]
    t = H[..., :3, 3]
    R_inv = R.transpose((0, 2, 1))
    t_inv = -np.matmul(R_inv, t[..., None])[..., 0]

    H_inv = np.eye(4, dtype=H.dtype)
    H_inv = np.repeat(H_inv[None], H.shape[0], axis=0)
    H_inv[..., :3, :3] = R_inv
    H_inv[..., :3, 3] = t_inv
    return H_inv


def get_pc_mask_per_H(H, pc):
    """
    H: B x 4 x 4
    pc: N x 3
    """
    H_inv = se3_inv_batched(H)
    pc = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=-1)

    # pc in H_inv frame
    pc = np.einsum("bmd,Nd->bNm", H_inv, pc)

    # get points within sv
    ctrl_pts = grasp_vis.get_gripper_control_pts()

    rect = ctrl_pts[[2, 5]]
    lat_delta = 0.02
    loc = ctrl_pts[-1, -1] - 0.005
    x_mask = (pc[..., 0] <= rect[0, 0]) & (pc[..., 0] >= rect[1, 0])
    y_mask = (pc[..., 1] <= lat_delta) & (pc[..., 1] >= -lat_delta)
    z_mask = (pc[..., 2] >= rect[0, 2]) & (pc[..., 2] <= loc + (loc - rect[0, 2]) / 5)
    pts_mask = x_mask & y_mask & z_mask
    return pts_mask  # B x N
