import numpy as np
import torch
from pytorch3d.transforms.so3 import so3_rotation_angle
from scipy.optimize import linear_sum_assignment


def empirical_emd(H_data, H_sample=None, n_grasps=500, episodes=1000) -> tuple:
    """
    Earth Mover's Distance (EMD) between two empirical distributions
    H_data corresponds to good grasps for an object
    H_sample corresponds to generated grasps
    - If None, then sample n_grasps from H_data
    """
    # Set Samples
    if H_sample is None:
        # Data-Data EMD
        idx = np.random.randint(0, H_data.shape[0], n_grasps)
        H_sample = torch.Tensor(H_data[idx, ...]).float()
    else:
        H_sample = torch.Tensor(H_sample).float()
    p_sample = H_sample[:, :3, -1]
    R_sample = H_sample[:, :3, :3]

    divergence = np.zeros(0, dtype=np.float32)
    for k in range(episodes):
        ## Sample Candidates ##
        idx = np.random.randint(0, H_data.shape[0], n_grasps)
        H_eval = torch.Tensor(H_data[idx, ...]).to(H_sample)

        p_eval = H_eval[:, :3, -1]
        R_eval = H_eval[:, :3, :3]

        # translation distance
        xyz_dist = (p_eval[:, None, ...] - p_sample[None, ...]).pow(2).sum(-1).pow(0.5)

        # rotation distance
        R12 = torch.einsum("bmn,knd->bkmd", R_eval.transpose(-1, -2), R_sample)
        R12_ = R12.reshape(-1, 3, 3)
        R_dist_ = 1.0 - so3_rotation_angle(R12_, cos_angle=True)
        R_dist = R_dist_.reshape(R12.shape[0], R12.shape[1])

        # total
        distance = xyz_dist + R_dist

        distance = distance.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(distance)
        min_distance = distance[row_ind, col_ind].mean()
        divergence = np.concatenate((divergence, np.array([min_distance])), axis=0)

    mean = np.mean(divergence)
    std = np.std(divergence)
    return mean, std
