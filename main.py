import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import os

from data import load_dataset
from model import VGG
from train import train, test
from quantization import apply_fake_quantization, QuantizedWrapper


def collect_features(model, dataloader, device, normalize=False):
    """Run the backbone to gather penultimate activations and labels."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)

            outputs = model.features(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            outputs = model.classifier(outputs)
            if normalize:
                outputs = F.normalize(outputs, p=2, dim=1)

            features.append(outputs.cpu())
            labels.append(targets)
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone_suffix = f"vgg11_{args.dataset.lower()}"
    backbone_path = f"checkpoint/{backbone_suffix}.pth"
    hd_suffix = None
    if args.use_hd_classifier:
        hd_suffix = f"{backbone_suffix}_hd{args.hd_dim}"
        if args.hd_disable_normalize:
            hd_suffix += "_nonorm"
        hd_checkpoint_path = f"checkpoint/{hd_suffix}.pth"
    else:
        hd_checkpoint_path = None

    model = VGG(
        'VGG11',
        num_classes=args.num_classes,
        use_hd_classifier=args.use_hd_classifier,
        hd_dim=args.hd_dim,
        hd_normalize=not args.hd_disable_normalize,
    ).to(device)
    if args.use_hd_classifier:
        model.hd_head.to(device)
    if args.use_hd_classifier:
        print(f"Using HD classifier head (dim={args.hd_dim}, normalize={not args.hd_disable_normalize})")
    train_loader, test_loader = load_dataset(args.batch_size, args.dataset)

    if os.path.exists(backbone_path):
        checkpoint = torch.load(backbone_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        load_result = model.load_state_dict(checkpoint, strict=not args.use_hd_classifier)
        print("Backbone loaded from", backbone_path)
        if hasattr(load_result, "missing_keys"):
            if load_result.missing_keys:
                print("Missing keys:", load_result.missing_keys)
            if load_result.unexpected_keys:
                print("Unexpected keys:", load_result.unexpected_keys)
    else:
        if args.use_hd_classifier:
            raise FileNotFoundError(
                f"Pretrained VGG checkpoint not found at {backbone_path}. "
                "Train the backbone first (run without --use_hd_classifier)."
            )
        print("Starting training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, criterion, optimizer, device, epoch)
            test(model, test_loader, device, epoch)
            scheduler.step()

        os.makedirs(os.path.dirname(backbone_path), exist_ok=True)
        torch.save(model.state_dict(), backbone_path)
        print("Backbone checkpoint saved to", backbone_path)

    hd_checkpoint = None
    if args.use_hd_classifier and os.path.exists(hd_checkpoint_path):
        print("HD checkpoint found at", hd_checkpoint_path, "- loading and skipping HD fitting.")
        hd_checkpoint = torch.load(hd_checkpoint_path, map_location=device)
    elif args.use_hd_classifier:
        model.eval()
        model.hd_head.to(device)
        model.hd_head.model.zero_()
        print("Collecting features from training set for OnlineHD fitting...")
        train_feats, train_labels = collect_features(
            model, train_loader, device, normalize=model.hd_normalize
        )
        print(f"Fitting OnlineHD classifier on {train_feats.size(0)} samples...")
        model.hd_head.fit(
            train_feats.to(device),
            train_labels.to(device),
            lr = args.lr,
            epochs = args.epochs,
            encoded=False,
            one_pass_fit=True,
            bootstrap=args.hd_bootstrap
        )
        # free large tensors early
        del train_feats, train_labels
        os.makedirs(os.path.dirname(hd_checkpoint_path), exist_ok=True)
        hd_checkpoint = {
            "backbone_state_dict": model.state_dict(),
            "hd_dim": args.hd_dim,
            "hd_normalize": model.hd_normalize,
            "num_classes": args.num_classes,
            "dataset": args.dataset,
            "hd_model": model.hd_head.model.detach().cpu(),
            "hd_encoder_basis": model.hd_head.encoder.basis.detach().cpu(),
            "hd_encoder_base": model.hd_head.encoder.base.detach().cpu(),
        }
        torch.save(hd_checkpoint, hd_checkpoint_path)
        print("HD checkpoint saved to", hd_checkpoint_path)
    elif args.use_hd_classifier:
        # hd_checkpoint is already populated from disk
        pass

    if args.use_hd_classifier and hd_checkpoint is not None:
        if "backbone_state_dict" in hd_checkpoint:
            model.load_state_dict(hd_checkpoint["backbone_state_dict"], strict=False)
        if "hd_model" in hd_checkpoint:
            model.hd_head.model.copy_(hd_checkpoint["hd_model"].to(device))
        if "hd_encoder_basis" in hd_checkpoint:
            model.hd_head.encoder.basis.copy_(hd_checkpoint["hd_encoder_basis"].to(device))
        if "hd_encoder_base" in hd_checkpoint:
            model.hd_head.encoder.base.copy_(hd_checkpoint["hd_encoder_base"].to(device))
        print("HD checkpoint loaded into model.")

    # -----------------------------
    # Fake quantization and testing
    # -----------------------------
    if args.network_quantization:
        print(f"Applying fake weight quantization ({args.network_quantization_bits} bits)...")
        model = apply_fake_quantization(model, args.network_quantization_bits)
        print("Re-running test with fake quantized weights...")
        test(model, test_loader, device, epoch='Quantized-Weights')

    if args.data_quantization:
        print(f"Applying fake data/activation quantization ({args.data_quantization_bits} bits)...")
        model = QuantizedWrapper(model, args.data_quantization_bits)
        print("Re-running test with fake quantized activations...")
        test(model, test_loader, device, epoch='Quantized-Data')

    noise_levels = torch.arange(0.0, 0.51, 0.01).tolist()
    original_state = model.state_dict()
    original_hd_model = None
    if args.use_hd_classifier:
        original_hd_model = model.hd_head.model.detach().clone()

    if args.noise_injection:
        print("Testing robustness under increasing weight noise...")
        for sigma in noise_levels:
            print(f"\n[Noise std: {sigma}] injecting into model weights...")
            if args.use_hd_classifier and original_hd_model is not None:
                noise = torch.randn_like(original_hd_model) * sigma
                model.hd_head.model.copy_(original_hd_model + noise)
                diff_norm = (model.hd_head.model - original_hd_model).norm().item()
                print(f"HD noise L2 delta: {diff_norm:.3f}")
            else:
                noisy_state = {}
                for name, param in original_state.items():
                    if 'weight' in name and param.dtype == torch.float32:
                        noise = torch.randn_like(param) * sigma
                        noisy_state[name] = param + noise
                    else:
                        noisy_state[name] = param.clone()
                model.load_state_dict(noisy_state)
            acc = test(model, test_loader, device, epoch=f'Noise-{sigma:.3f}')
            print(f"Accuracy with noise std={sigma:.3f}: {acc:.2f}%")
    if args.use_hd_classifier and original_hd_model is not None:
        model.hd_head.model.copy_(original_hd_model)
    else:
        model.load_state_dict(original_state)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG11 CIFAR10 Training')

    # Dataset & optimization -------------------------------------------------
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'], help='Dataset to use (default: CIFAR10)')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training and testing')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--step_size', default=20, type=int, help='Step size for LR scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma for LR scheduler')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--noise_injection', action='store_true', help='test model robustbess with diff noise level')

    # Quantization knobs ----------------------------------------------------
    parser.add_argument('--network_quantization', action='store_true', help='Enable network weight quantization to 4 bits')
    parser.add_argument('--network_quantization_bits', default=4, type=int, help='Number of bits for network weight quantization')
    parser.add_argument('--data_quantization', action='store_true', help='Enable data activation quantization to 5 bits')
    parser.add_argument('--data_quantization_bits', default=5, type=int, help='Number of bits for data activation quantization')

    # HD classifier options -------------------------------------------------
    parser.add_argument('--use_hd_classifier', action='store_true', help='Replace final linear layer with HD classifier head')
    parser.add_argument('--hd_dim', default=10000, type=int, help='Dimensionality of the HD classifier head')
    parser.add_argument('--hd_disable_normalize', action='store_true', help='Disable L2-normalization inside the HD classifier head')
    parser.add_argument('--hd_bootstrap', default=0.01, type=float, help='Bootstrap fraction (0,1] used to seed HD hypervectors')
    parser.add_argument('--hd_one_pass_fit', dest='hd_one_pass_fit', action='store_true', help='Enable one-pass HD initialization before iterative fitting')
    parser.add_argument('--hd_skip_one_pass_fit', dest='hd_one_pass_fit', action='store_false', help='Disable one-pass HD initialization')
    parser.set_defaults(hd_one_pass_fit=True)

    args = parser.parse_args()
    args.num_classes = 10 if args.dataset == 'CIFAR10' else 100
    main(args)
