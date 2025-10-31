# VGG + OnlineHD CIFAR Classifier

## Overview
- VGG11 backbone for CIFAR-10/100 with standard SGD training.
- Optional replacement of the final linear layer with an OnlineHD hyperdimensional classifier.
- Post-training experiments: fake quantization of weights/activations and robustness sweeps with injected Gaussian noise.
- Automatic checkpoint management for both backbone-only and HD-augmented models.

## Project Layout
- `main.py` – entry point; orchestrates training, feature extraction, HD fitting, quantization, and noise evaluation.
- `model.py` – VGG11 definition with optional OnlineHD head.
- `data.py` – CIFAR dataset loaders and augmentation pipelines.
- `train.py` – standard train/eval loops (tqdm progress bars if installed).
- `quantization.py` – helper utilities for fake quantization and activation wrappers.
- `onlinehd/` – OnlineHD implementation plus C++ acceleration (`fasthd`) compiled on first import.
- `checkpoint/` – saved backbone and HD checkpoints.
- `data/` – CIFAR datasets downloaded automatically by `torchvision`.

## Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12 and torchvision
- tqdm (optional, for progress bars)
- A C++14 toolchain (e.g., GCC or Clang) for compiling `onlinehd/fasthd` via `torch.utils.cpp_extension`

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision tqdm
```

The first import of `onlinehd` will trigger a JIT build of the C++ extension; ensure the toolchain is available on your system.

## Training the VGG Backbone

```bash
cd vgg_code
python main.py --dataset CIFAR10
```

- Supports `--dataset CIFAR10` or `CIFAR100`.
- Default hyperparameters: `epochs=50`, `lr=0.01`, `momentum=0.9`, `weight_decay=5e-4`, `batch_size=128`.
- The trained backbone is saved as `checkpoint/vgg11_<dataset>.pth`.
- If the checkpoint already exists, it is loaded instead of retrained.

## Adding the OnlineHD Classifier

After training the backbone, fit the hyperdimensional head:

```bash
python main.py --dataset CIFAR10 --use_hd_classifier --hd_dim 10000
```

What happens:
- Loads the backbone checkpoint (strict=False to allow the missing final linear layer).
- Runs the backbone in eval mode to collect penultimate features and labels.
- Fits OnlineHD (`dim=10000` by default) with optional feature normalization (enabled unless `--hd_disable_normalize` is set).
- Saves an HD-specific checkpoint at `checkpoint/vgg11_<dataset>_hd<dim>[_nonorm].pth` containing:
  - Backbone weights,
  - OnlineHD class hypervectors,
  - Encoder basis/base tensors,
  - Metadata (dataset, number of classes, HD dimension).
- Subsequent runs with the same flags detect the HD checkpoint, load it, and jump straight to evaluation.

## Quantization Experiments

Two command-line toggles let you mimic lower-precision hardware:

- `--network_quantization --network_quantization_bits 4`  
  Permanently fake-quantizes every parameter tensor (values land on the 4-bit grid) before evaluation. Saving the model after this step gives you a weight-compressed checkpoint.

- `--data_quantization --data_quantization_bits 5`  
  Wraps the model so that activations are fake-quantized to the requested bit width. Every ReLU output inside the network (not just the model I/O) is rounded to that grid, matching our 5-bit activation constraint.

Combine the two flags to replicate the current hardware plan (4-bit weights, 5-bit activations).

## Noise Robustness Sweeps

Noise experiments now live in `scripts/`:

- `scripts/sweep_vgg_noise.sh` runs the backbone with optional 4-bit weight + 5-bit activation quantization and sweeps Gaussian activation-noise sigmas. Pass `--output logs/foo.csv` to archive results.
- `scripts/sweep_vgg_hd64_noise.sh` mirrors the workflow for HD checkpoints (default `hd_dim=64`).

Example (hardware-mimicking sweep):

```bash
conda run -n vgg_hd bash scripts/sweep_vgg_noise.sh \
  --sigma-start 0.0 --sigma-stop 0.5 --sigma-step 0.01 \
  --weight-bits 4 --bits 5 \
  --output logs/vgg11_noise_quant.csv
```

The CSV logs contain `sigma,accuracy` pairs you can plot or compare against previous runs.

## Checkpoint Summary

| Purpose                 | Path pattern                                 | Notes                                                      |
|-------------------------|----------------------------------------------|------------------------------------------------------------|
| VGG backbone            | `checkpoint/vgg11_<dataset>.pth`             | Created during standard training.                          |
| OnlineHD classifier     | `checkpoint/vgg11_<dataset>_hd<dim>[_nonorm].pth` | Written after fitting HD; reused on subsequent runs.       |

Delete or rename HD checkpoints to force a refit with different hyperparameters.

## Tips & Troubleshooting
- If PyTorch cannot compile the `onlinehd` extension, ensure build essentials are installed (e.g., `sudo apt install build-essential` on Debian/Ubuntu).
- HD fitting loads the entire training set into memory to concatenate features; lower `batch_size` or modify `collect_features` if this becomes a constraint.
- To experiment with different noise profiles, edit the `noise_levels` range in `main.py`.
- CIFAR datasets download to `vgg_code/data` automatically; delete the folder to refresh the cache.

## Citation / Attribution
- OnlineHD is based on the implementation bundled in this repository (`onlinehd` folder) and references the paper “Online Hyperdimensional Learning for Classifying Data Streams” (Imani et al., 2019).
