import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    data_iter = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False) if tqdm else train_loader
    for batch_idx, (inputs, targets) in enumerate(data_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if tqdm:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100.0 * correct / total if total else 0.0
            data_iter.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{acc:.2f}%")

    print(f"[Epoch {epoch}] Train Acc: {100. * correct / total:.2f}%, Loss: {running_loss / len(train_loader):.3f}")


def test(model, test_loader, device, epoch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        data_iter = tqdm(test_loader, desc=f"Epoch {epoch} [Eval]", leave=False) if tqdm else test_loader
        for inputs, targets in data_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if tqdm:
                acc = 100.0 * correct / total if total else 0.0
                data_iter.set_postfix(acc=f"{acc:.2f}%")

    print(f"[Epoch {epoch}] Test  Acc: {100. * correct / total:.2f}%")
    return 100. * correct / total
