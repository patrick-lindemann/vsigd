# Variational Shape Inference for Grasp Diffusion on SE(3)

<!-- Shields-->
[<img src="https://img.shields.io/badge/arXiv-%230077B5.svg?&style=plastic&logo=arxiv&labelColor=ff0000&color=ffffff" />](https://arxiv.org/abs/2508.17482)
[<img src="https://img.shields.io/badge/Paper-%230077B5.svg?&style=plastic&logo=ieee&labelColor=00629b&color=ffffff" />](https://ieeexplore.ieee.org/document/11302796)
[<img src="https://img.shields.io/badge/Demos-%230077B5.svg?&style=plastic&logo=youtube&labelColor=cd201f&color=ffffff" />](https://youtu.be/YyJPZQgEErM?si=zEaETt5C12FjFMXF)
[<img src="https://img.shields.io/badge/Data-%230077B5.svg?&style=plastic&logo=google-drive&labelColor=white&color=blue" />](https://drive.google.com/drive/folders/1xV4fJjdUIpxFVsEOBQm32DzecbWMUBrZ?usp=sharing)

## Summary

We propose a framework for learning multimodal grasp distributions that leverages variational shape inference to enhance robustness against shape noise and measurement sparsity. Our approach learns a variational autoencoder for shape inference using implicit neural representations, and then uses these learned geometric features to guide a diffusion model for grasp synthesis on the SE(3) manifold. Additionally, we introduce a test-time grasp optimization technique that can be integrated as a plugin to further enhance grasping performance. Experimental results demonstrate that our shape inference for grasp synthesis formulation outperforms state-of-the-art multimodal grasp synthesis methods on the [ACRONYM](https://github.com/NVlabs/acronym) dataset by 6.3%, while demonstrating robustness to deterioration in point cloud density compared to other approaches. Furthermore, our trained model achieves zero-shot transfer to real-world manipulation of household objects, generating 34% more successful grasps than baselines despite measurement noise and point cloud calibration errors.

## Setup

Tested with PyTorch `2.0.1` + CUDA `11.7`

```sh
conda create -y -n vsigd python=3.8
conda activate vsigd
source pkgs-install.sh
```

(Optional) For headless rendering:

```sh
sudo apt-get install mesa-common-dev -y
conda install -y pyopengl pyopengl-accelerate mesalib pyrender -c conda-forge
# Embree needed for faster rendering of partial point clouds:
conda install -y pyembree -c conda-forge
```

## Testing

Pretrained checkpoints can be downloaded from [***here***](https://drive.google.com/drive/folders/1xV4fJjdUIpxFVsEOBQm32DzecbWMUBrZ?usp=sharing).

```sh
# on acronym dataset
python infer.py \
  --cfg-dir "path/to/train-session" \
  --ckpt-filename "ckpt.pth"
# on arbitrary point clouds
python infer_rw.py \
  --cfg-dir "path/to/train-session" \
  --ckpt-filename "ckpt.pth" \
  --object-pc "pc.npy"
```

## Preparation

Grasps and corresponding processed meshes from the [ACRONYM](https://github.com/NVlabs/acronym) dataset are available [***here***](https://drive.google.com/drive/folders/1xV4fJjdUIpxFVsEOBQm32DzecbWMUBrZ?usp=sharing).

Pre-sample points on meshes and SDF point-value pairs using `src/main.py` from the [`mesh2sdf-cuda`](https://github.com/stalhabukhari/mesh2sdf-cuda) repository:

```sh
python main.py \
    --dataset_dir "path/to/meshes" \
    --save_dir "path/to/mesh2sdf-600k" \
    --num_samples_surf 100000 \
    --num_samples_sdf 600000 \
    --chunk_size 100000
```

See [`data-dir/README.md`](data-dir/README.md) for the expected directory structure for the dataset.
Save directory name `mesh2sdf-600k` corresponds to the `data: sdf_data_subdir` field in the training configuration file.

Also place pretrained PointVAE checkpoints under `ckpts-ptvae/`.

## Training

```sh
python train.py --cfg configs/train_<PCMODE>.yml
# where PCMODE in [full, partial]
```

## Citation

```bibtex
@article{bukhari25ral,
  title={Variational Shape Inference for Grasp Diffusion on SE(3)},
  author={Bukhari, S. Talha and Agrawal, Kaivalya and Kingston, Zachary and Bera, Aniket},
  booktitle={Robotics and Automation Letters (RA-L)},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement

We thank the authors of the following repositories, which we adapt code from:
- https://github.com/princeton-computational-imaging/Diffusion-SDF
- https://github.com/robotgradient/grasp_diffusion
- https://github.com/bdlim99/EquiGraspFlow

## License

Code is released under the MIT License. See the LICENSE file for more details.
