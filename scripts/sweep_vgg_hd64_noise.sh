#!/usr/bin/env bash

# Sweep Gaussian activation-noise sigmas for the VGG backbone with HD head (dim=64 by default).
# Usage mirrors sweep_vgg_noise.sh but expects an HD checkpoint bundle.

set -euo pipefail

usage() {
    cat <<'EOF'
Sweep Gaussian activation-noise sigmas for VGG + HD classifier (default hd_dim=64).

Options:
  --sigmas "s0 s1 ..."    Explicit whitespace-separated sigma list.
  --sigma-start VALUE     Inclusive sigma range start (default: 0.0).
  --sigma-stop VALUE      Inclusive sigma range stop  (default: 1.0).
  --sigma-step VALUE      Step between sigmas         (default: 0.1).
  --bits N                Activation quantization bits (default: 5).
  --weight-bits N         Weight quantization bits (default: 4).
  --batch-size N          Evaluation batch size        (default: 128).
  --dataset NAME          CIFAR10 or CIFAR100          (default: CIFAR10).
  --checkpoint PATH       HD checkpoint bundle path    (default: checkpoint/vgg11_cifar10_hd64.pth).
  --hd-dim N              HD dimensionality            (default: 64).
  --output PATH           Optional CSV output path.
  --cpu                   Force CPU execution.
  --no-eval-noise         Disable noise when the model is in eval mode.
  -h, --help              Show this message.

Examples:
  ./sweep_vgg_hd64_noise.sh --sigma-start 0 --sigma-stop 0.5 --sigma-step 0.05
  ./sweep_vgg_hd64_noise.sh --sigmas "0 0.1 0.25 0.5" --output logs/hd64_noise.csv
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
CHECKPOINT="checkpoint/vgg11_cifar10_hd64.pth"
OUTPUT=""
FORCE_CPU="0"
NOISE_DURING_EVAL="1"
HD_DIM="64"

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
        --hd-dim)
            shift
            HD_DIM="${1:?Missing value for --hd-dim}"
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
export SWEEP_HD_DIM="${HD_DIM}"

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
hd_dim = int(os.environ["SWEEP_HD_DIM"])

device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

train.tqdm = None
_, test_loader = load_dataset(batch_size, dataset)

bundle = torch.load(checkpoint_path, map_location="cpu")
required_keys = {"backbone_state_dict", "hd_model", "hd_encoder_basis", "hd_encoder_base"}
missing = required_keys - set(bundle.keys())
if missing:
    raise KeyError(f"Checkpoint bundle missing keys: {missing}")

results: list[tuple[float, float]] = []

print(f"Device: {device}")
print(f"Checkpoint: {checkpoint_path}")
print(f"Weight bits: {weight_bits}")
print(f"Activation bits: {bits}")
print(f"HD dim: {hd_dim}")
print(f"Noise during eval: {noise_eval}")
print("sigma,accuracy")

for sigma in sigmas:
    model = VGG(
        use_hd_classifier=True,
        hd_dim=hd_dim,
        hd_normalize=bundle.get("hd_normalize", True),
        activation_noise=True,
        activation_noise_bits=bits,
        activation_noise_sigma=sigma,
        activation_noise_eval=noise_eval,
    )
    model.load_state_dict(bundle["backbone_state_dict"], strict=False)
    model.to(device)
    model.hd_head.to(device)
    model.hd_head.model.copy_(bundle["hd_model"].to(device))
    model.hd_head.encoder.basis.copy_(bundle["hd_encoder_basis"].to(device))
    model.hd_head.encoder.base.copy_(bundle["hd_encoder_base"].to(device))
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
