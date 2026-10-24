# PG-DUN for Snapshot 3D Imaging

PyTorch implementation of **Physics-guided deep unfolding network for snapshot 3D imaging using double-helix point spread function**, published in *Advanced Photonics Nexus* (2026).

Gang Qu, Pengwei Wang, Xing Liu, Haomiao Zhang, Mengyuan Liu, Zhentao Liu, and Xin Yuan.

[Paper](https://doi.org/10.1117/1.APN.5.5.056024)

## Overview

PG-DUN combines a calibrated double-helix point spread function (DH-PSF) imaging model with a deep unfolding network for snapshot 3D reconstruction. The network alternates physics-based deconvolutional gradient updates with learned reconstruction modules to recover depth-resolved image slices. This release includes the network, training and evaluation scripts, ten calibrated PSF images, and a pretrained model.

## Installation

Use Linux with an NVIDIA GPU and a compatible driver. The following reference setup uses Python 3.13.2 and package versions from the working `SPI` environment; it is not a full training-environment export.

Install [Git LFS](https://git-lfs.com/) before downloading the pretrained model:

```bash
git lfs install
git clone https://github.com/Gang-Qu/PGDUN_snapshot_3D_imaging.git
cd PGDUN_snapshot_3D_imaging
git lfs pull
conda create -n pgdun python=3.13.2 -y
conda activate pgdun
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

The dependency file selects the CUDA 12.6 PyTorch build. CUDA is required by the current PSF loader.

## Repository Structure

```text
PGDUN_snapshot_3D_imaging/
  Calibrated_PSFs/             # Ten depth-dependent calibration images
  model/Network.py            # PGDUN architecture and imaging operators
  weight/Pretrained_Weight.pth # Pretrained model, stored using Git LFS
  opts.py                     # Model, dataset, and training configuration
  train.py                    # Training
  test.py                     # Evaluation and reconstruction export
  train.sh                    # Example distributed-training launcher
  para_Flops.py               # Profiling script
  utils.py                    # Data loaders, metrics, and utilities
  requirement.txt             # Reference dependencies
```

The local `dataaset/` and generated `test_results/` directories are not included.

## Data and Calibration

Prepare separate training and test directories containing MATLAB `.mat` files. Each file must contain a `sliced_image` array with shape `H x W x C`. The default configuration uses ten depth slices and `800 x 800` spatial dimensions. The loaders convert arrays to float32 and divide intensities by 255, so prepare data on the corresponding 0-255 intensity scale.

Before running, replace both `/path/to/Calibrated_PSFs` placeholders inside `load_psf()` in `model/Network.py` with the absolute path to this repository's `Calibrated_PSFs` directory. The loader sorts the calibration filenames and applies the supplied fixed crop for the `800 x 800` setup. Keep the PSF order aligned with the depth-channel order in the dataset.

Edit `opts.py` for list-valued image dimensions and architecture settings. The default model has ten channels and six unfolding stages; evaluation also uses fixed `800 x 800` visualization dimensions.

## Testing

Run from the repository root after configuring the PSF path:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --test_weight_path ./weight/Pretrained_Weight.pth \
  --benchmark_path /path/to/test-dataset \
  --device cuda
```

The script reports projected-image PSNR/SSIM and slice-wise reconstruction error. Logs are saved in `test_results/log/`; projected-image and depth-slice comparisons are saved in `test_results/<decoder_type>/0115/`.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_data_path /path/to/train-dataset \
  --benchmark_path /path/to/test-dataset \
  --device cuda
```

Training outputs are written to `weights/`, `log/`, and `results/`. The supplied `train.sh` is a machine-specific distributed-training example; update its project path, Conda environment, and GPU selection before use.

## Citation

```bibtex
@article{qu2026physics,
  title   = {Physics-guided deep unfolding network for snapshot 3D imaging using double-helix point spread function},
  author  = {Qu, Gang and Wang, Pengwei and Liu, Xing and Zhang, Haomiao and Liu, Mengyuan and Liu, Zhentao and Yuan, Xin},
  journal = {Advanced Photonics Nexus},
  volume  = {5},
  number  = {5},
  pages   = {056024},
  year    = {2026},
  doi     = {10.1117/1.APN.5.5.056024}
}
```
