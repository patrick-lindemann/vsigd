from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization as vis


def mesh_obj_colorize_o3d_(scene_mesh):
    try:
        _colors = np.array([128, 0, 128], dtype=np.float32) / 255.0
        scene_mesh.paint_uniform_color(_colors)  # Open3D supports RGB, not RGBA
        scene_mesh.compute_triangle_normals()
        scene_mesh.compute_vertex_normals()

    except Exception as e:
        print(f"Error in setting mesh color: {e}")


def pc_to_o3d(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # colorize
    z = pc[:, 2].reshape(-1, 1).copy()
    z = (z - z.min()) / (z.max() - z.min())
    colors = (1 - z) * np.array([255, 0, 0])[None, :] + z * np.array([0, 0, 255])[
        None, :
    ]
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return pcd


def get_gripper_control_pts(scale=1.0):
    """
    6-point gripper parameterization
    """
    midz = 0.0659999996 * scale
    endz = 0.11217 * scale
    x_off = 0.041 * scale
    pts = [
        [0, 0, 0],  # origin is important
        [0, 0, midz],  # center
        [x_off, 0, midz],  # left gripper stem
        [-x_off, 0, midz],  # right gripper stem
        [x_off, 0, endz],  # left gripper tip
        [-x_off, 0, endz],  # right gripper tip
    ]
    pts = np.array(pts, dtype=np.float32)
    return pts


def create_gripper_marker_o3d(color=[0, 0, 255], sections=6, scale=1.0):
    """Create a 3D mesh visualizing a parallel yaw gripper using Open3D. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        o3d.geometry.TriangleMesh: A mesh that represents a simple parallel yaw gripper.
    """

    def create_cylinder(radius, height, resolution, translation):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=height, resolution=resolution
        )
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(np.array(color) / 255.0)
        cylinder.translate(translation)
        return cylinder

    # Create cylinders for the gripper
    cfl = create_cylinder(
        0.002 * scale,
        0.11217 * scale - 0.0659999996 * scale,
        sections,
        [0.041 * scale, 0, (0.0659999996 + 0.11217) / 2 * scale],
    )  # left clip
    cfr = create_cylinder(
        0.002 * scale,
        0.11217 * scale - 0.0659999996 * scale,
        sections,
        [-0.041 * scale, 0, (0.0659999996 + 0.11217) / 2 * scale],
    )  # right clip
    cb1 = create_cylinder(
        0.002 * scale,
        0.0659999996 * scale,
        sections,
        [0, 0, 0.0659999996 / 2 * scale],
    )  # rear handle
    cb2 = create_cylinder(
        0.002 * scale,
        0.082 * scale,
        sections,
        [0, 0, 0.0659999996 * scale],
    )  # crossbar
    cb2.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]), center=[0, 0, 0]
    )
    cb2.translate([0, 0, 0.0659999996 * scale], relative=False)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)

    # Combine all parts into a single mesh
    gripper = cfl + cfr + cb1 + cb2
    gripper = gripper + axes
    return gripper


def get_scene_geoms_o3d(
    pc_surf: Optional[np.ndarray] = None,
    mesh_obj: Optional[o3d.geometry.TriangleMesh] = None,
    pc_surf_mean: Optional[np.ndarray] = None,
    rand_H: Optional[np.ndarray] = None,
    H_grasps_gt: Optional[np.ndarray] = None,
    H_grasps_pred: Optional[np.ndarray] = None,
    H_grasps_pred_traj: Optional[np.ndarray] = None,
    dataset_scale: float = 1.0,
    gripper_scale: float = 1.0,
):
    # context (pc)
    if pc_surf is not None:
        pc_surf = pc_to_o3d(pc_surf)

    # mesh (for vis)
    if mesh_obj is not None:
        mesh_obj.scale(dataset_scale, center=[0, 0, 0])
        mesh_obj.translate(-pc_surf_mean, relative=True)
        if rand_H is not None:
            mesh_obj.transform(rand_H)
        mesh_obj.translate(pc_surf_mean, relative=True)
        mesh_obj.scale(1 / dataset_scale, center=[0, 0, 0])

        mesh_obj_colorize_o3d_(mesh_obj)

    # gripper gt
    grasp_gt_list = []
    if H_grasps_gt is not None:
        for i in range(H_grasps_gt.shape[0]):
            _grasp_gt = create_gripper_marker_o3d(
                scale=gripper_scale, color=[0, 255, 0]  # green
            )
            _grasp_gt.transform(H_grasps_gt[i])
            grasp_gt_list.append({f"grasp-gt-{i}": _grasp_gt})

    # gripper pred
    grasp_pred_list = []
    if H_grasps_pred is not None:
        for i in range(H_grasps_pred.shape[0]):
            # gripper
            _grasp_pred = create_gripper_marker_o3d(
                scale=gripper_scale, color=[0, 0, 255]  # blue
            )
            _grasp_pred.transform(H_grasps_pred[i])
            grasp_pred_list.append({f"grasp-pred-{i}": _grasp_pred})
            # TODO: add option for grasp axes

    grasp_pred_traj_list = []
    if H_grasps_pred_traj is not None:
        z = np.linspace(0, 1, H_grasps_pred_traj.shape[0])[:, None].astype(np.float32)
        colors = (1 - z) * np.array([1.0, 0.0, 0.0])[None, :] + z * np.array(
            [0.0, 0.0, 1.0]
        )[None, :]
        colors = colors.astype(np.float32)

        for t in range(H_grasps_pred_traj.shape[0]):  # last element already plotting
            H_grasps_pred_i = H_grasps_pred_traj[t]
            for i in range(H_grasps_pred_i.shape[0]):
                # gripper
                _c = colors[t] * 255
                if t == H_grasps_pred_traj.shape[0] - 1:  # final
                    _c = [0.0, 255.0, 0.0]
                _grasp_pred_traj = create_gripper_marker_o3d(
                    scale=gripper_scale,
                    color=_c,
                )
                _grasp_pred_traj.transform(H_grasps_pred_i[i])
                grasp_pred_traj_list.append({f"grasp-pred-traj-{i}": _grasp_pred_traj})

    scene_geoms = [
        # model input:
        *grasp_gt_list,
        # model output:
        *grasp_pred_list,
        # *grasp_axes_list,
        *grasp_pred_traj_list,
    ]
    if mesh_obj is not None:
        scene_geoms.append({"object": mesh_obj})
    if pc_surf is not None:
        scene_geoms.append({"pc": pc_surf})
    return scene_geoms


def render_write_o3d(geoms, out_path):
    # geoms is a list of key, value pairs. ref: get_scene_geoms_o3d()

    # Setup renderer
    h, w = 1080, 1920
    render = o3d.visualization.rendering.OffscreenRenderer(width=w, height=h)

    # define material (TODO: move to get_scene_geoms_o3d)
    mtrl = vis.rendering.MaterialRecord()
    mtrl.shader = "defaultLitTransparency"
    mtrl.base_roughness = 0.01
    mtrl.base_reflectance = 0.25
    mtrl.thickness = 1.0
    mtrl.transmission = 0.25

    # add geometries to scene
    for idx, geom in enumerate(geoms):
        for k, v in geom.items():
            render.scene.add_geometry(k, v, mtrl)
    render.scene.set_background([0.8, 0.8, 0.8, 1.0])

    # set the camera to fit object view
    center = np.array(render.scene.bounding_box.get_center(), dtype=np.float32)
    half = np.array(render.scene.bounding_box.get_half_extent(), dtype=np.float32)

    # re-center geometries
    for geom in geoms:
        for k, v in geom.items():
            geom_tf = render.scene.get_geometry_transform(k)
            geom_tf[:3, -1] -= center
            render.scene.set_geometry_transform(k, geom_tf)
    center = np.array([0, 0, 0], dtype=np.float32)

    low = min(center - half) * 1.25
    high = max(center + half) * 1.25

    aspect = h / w
    # s = 3
    render.scene.camera.set_projection(
        vis.rendering.Camera.Projection.Ortho,
        low,
        high,
        low * aspect,
        high * aspect,
        0.1,
        200,
    )
    # render.scene.camera.look_at([0, 0, 0], [100, 100, 100], [0, 0, 1])
    half *= 1.5
    render.scene.camera.look_at(
        center,
        center + half,
        [-1, 0, 0],
    )

    # Render and write to filepath
    img = render.render_to_image()
    o3d.io.write_image(str(out_path), img)


if __name__ == "__main__":
    # testing
    gripper = create_gripper_marker_o3d(scale=8.0)
    # o3d.visualization.draw_geometries([gripper])

    # on hpc
    output_path = "render-gripper.png"
    render_write_o3d([gripper], output_path)
