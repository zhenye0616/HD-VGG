#!/usr/bin/env bash

# Sweep Gaussian activation-noise sigmas for the baseline VGG backbone.
# Usage:
#   ./sweep_vgg_noise.sh [--sigmas "0 0.1 0.2"] [--sigma-start 0.0 --sigma-stop 1.0 --sigma-step 0.1]
#                         [--bits 5] [--weight-bits 4] [--batch-size 128] [--dataset CIFAR10]
#                         [--checkpoint checkpoint/vgg11_cifar10.pth] [--output results.csv]
#                         [--cpu] [--no-eval-noise]
#
# Requires an environment where `python` resolves to the desired interpreter
# with torch/torchvision installed (e.g., `conda activate vgg_hd`).

set -euo pipefail

usage() {
    cat <<'EOF'
Sweep Gaussian activation-noise sigmas for the baseline VGG backbone.

Options:
  --sigmas "s0 s1 ..."   Explicit whitespace-separated sigma list.
  --sigma-start VALUE    Inclusive sigma range start (default: 0.0).
  --sigma-stop VALUE     Inclusive sigma range stop  (default: 1.0).
  --sigma-step VALUE     Step between sigmas         (default: 0.1).
  --bits N               Activation quantization bits (default: 5).
  --weight-bits N        Weight quantization bits (default: 4).
  --batch-size N         Evaluation batch size        (default: 128).
  --dataset NAME         CIFAR10 or CIFAR100          (default: CIFAR10).
  --checkpoint PATH      Baseline checkpoint path     (default: checkpoint/vgg11_cifar10.pth).
  --output PATH          Optional CSV output path.
  --cpu                  Force CPU execution.
  --no-eval-noise        Disable noise when the model is in eval mode.
  -h, --help             Show this message.

Examples:
  ./sweep_vgg_noise.sh --sigma-start 0 --sigma-stop 0.5 --sigma-step 0.05
  ./sweep_vgg_noise.sh --sigmas "0 0.1 0.25 0.5" --output logs/vgg_noise.csv
EOF
}

SIGMA_LIST=""
SIGMA_START="0.0"
SIGMA_STOP="1.0"
SIGMA_STEP="0.1"
BITS="5"
WEIGHT_BITS="4"
BATCH_SIZE="128"
DATASET="CIFAR10"
CHECKPOINT="checkpoint/vgg11_cifar10.pth"
OUTPUT=""
FORCE_CPU="0"
NOISE_DURING_EVAL="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sigmas)
            shift
            SIGMA_LIST="${1:-}"
            ;;
        --sigma-start)
            shift
            SIGMA_START="${1:?Missing value for --sigma-start}"
            ;;
        --sigma-stop)
            shift
            SIGMA_STOP="${1:?Missing value for --sigma-stop}"
            ;;
        --sigma-step)
            shift
            SIGMA_STEP="${1:?Missing value for --sigma-step}"
            ;;
        --bits)
            shift
            BITS="${1:?Missing value for --bits}"
            ;;
        --batch-size)
            shift
            BATCH_SIZE="${1:?Missing value for --batch-size}"
            ;;
        --dataset)
            shift
            DATASET="${1:?Missing value for --dataset}"
            ;;
        --checkpoint)
            shift
            CHECKPOINT="${1:?Missing value for --checkpoint}"
            ;;
        --output)
            shift
            OUTPUT="${1:?Missing value for --output}"
            ;;
        --weight-bits)
            shift
            WEIGHT_BITS="${1:?Missing value for --weight-bits}"
            ;;
        --cpu)
            FORCE_CPU="1"
            ;;
        --no-eval-noise)
            NOISE_DURING_EVAL="0"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift || true
done

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

export SWEEP_SIGMA_LIST="${SIGMA_LIST}"
export SWEEP_SIGMA_START="${SIGMA_START}"
export SWEEP_SIGMA_STOP="${SIGMA_STOP}"
export SWEEP_SIGMA_STEP="${SIGMA_STEP}"
export SWEEP_BITS="${BITS}"
export SWEEP_WEIGHT_BITS="${WEIGHT_BITS}"
export SWEEP_BATCH="${BATCH_SIZE}"
export SWEEP_DATASET="${DATASET}"
export SWEEP_CHECKPOINT="${CHECKPOINT}"
export SWEEP_OUTPUT="${OUTPUT}"
export SWEEP_FORCE_CPU="${FORCE_CPU}"
export SWEEP_NOISE_EVAL="${NOISE_DURING_EVAL}"

python - <<'PY'
import os
from pathlib import Path

import torch

import train
from data import load_dataset
from model import VGG
from quantization import QuantizedWrapper, apply_fake_quantization


def parse_sigmas() -> list[float]:
    explicit = os.environ.get("SWEEP_SIGMA_LIST", "").strip()
    if explicit:
        return [float(s) for s in explicit.split()]

    start = float(os.environ["SWEEP_SIGMA_START"])
    stop = float(os.environ["SWEEP_SIGMA_STOP"])
    step = float(os.environ["SWEEP_SIGMA_STEP"])
    if step <= 0:
        raise ValueError("--sigma-step must be positive.")

    sigmas = []
    value = start
    for _ in range(10000):
        if value > stop + 1e-9:
            break
        sigmas.append(round(value, 10))
        value += step
    if not sigmas:
        raise ValueError("No sigma values generated; check sigma range.")
    return sigmas


sigmas = parse_sigmas()
bits = int(os.environ["SWEEP_BITS"])
weight_bits = int(os.environ["SWEEP_WEIGHT_BITS"])
batch_size = int(os.environ["SWEEP_BATCH"])
dataset = os.environ["SWEEP_DATASET"]
checkpoint_path = Path(os.environ["SWEEP_CHECKPOINT"])
output_path = os.environ.get("SWEEP_OUTPUT", "").strip()
force_cpu = os.environ.get("SWEEP_FORCE_CPU", "0") == "1"
noise_eval = os.environ.get("SWEEP_NOISE_EVAL", "1") == "1"

device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

train.tqdm = None
_, test_loader = load_dataset(batch_size, dataset)

state_dict = torch.load(checkpoint_path, map_location="cpu")
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

results: list[tuple[float, float]] = []

print(f"Device: {device}")
print(f"Checkpoint: {checkpoint_path}")
print(f"Weight bits: {weight_bits}")
print(f"Activation bits: {bits}")
print(f"Noise during eval: {noise_eval}")
print("sigma,accuracy")

for sigma in sigmas:
    model = VGG(
        activation_noise=True,
        activation_noise_bits=bits,
        activation_noise_sigma=sigma,
        activation_noise_eval=noise_eval,
    )
    model.load_state_dict(state_dict, strict=True)
    apply_fake_quantization(model, num_bits=weight_bits)
    model = QuantizedWrapper(model, bits)
    model.to(device)
    model.eval()
    acc = train.test(model, test_loader, device, epoch=f"sigma={sigma:.3f}")
    results.append((sigma, acc))
    print(f"{sigma:.4f},{acc:.4f}")

if output_path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("sigma,accuracy\n")
        for sigma, acc in results:
            f.write(f"{sigma:.6f},{acc:.6f}\n")
    print(f"Saved results to {output_file}")
PY
