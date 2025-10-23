import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import os

from model import VGG
from data import load_dataset
from train import train, test
from quantization import apply_fake_quantization, QuantizedWrapper


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG('VGG11', num_classes=args.num_classes).to(device)
    train_loader, test_loader = load_dataset(args.batch_size, args.dataset)

    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path))
        print("Model loaded from", args.save_path)
    else:
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

        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)

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

    print("Testing robustness under increasing weight noise...")
    noise_levels = torch.arange(0.0, 0.21, 0.01).tolist()
    original_state = model.state_dict()

    for sigma in noise_levels:
        print(f"\n[Noise std: {sigma}] injecting into model weights...")
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
    model.load_state_dict(original_state)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG11 CIFAR10 Training')
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'], help='Dataset to use (default: CIFAR10)')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training and testing')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--step_size', default=20, type=int, help='Step size for LR scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma for LR scheduler')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')

    parser.add_argument('--network_quantization', action='store_true', help='Enable network weight quantization to 4 bits')
    parser.add_argument('--network_quantization_bits', default=4, type=int, help='Number of bits for network weight quantization')

    parser.add_argument('--data_quantization', action='store_true', help='Enable data activation quantization to 5 bits')
    parser.add_argument('--data_quantization_bits', default=5, type=int, help='Number of bits for data activation quantization')

    args = parser.parse_args()
    args.save_path = f"checkpoint/vgg11_{args.dataset.lower()}.pth"
    args.num_classes = 10 if args.dataset == 'CIFAR10' else 100
    main(args)