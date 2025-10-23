import torch


def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

    print(f"[Epoch {epoch}] Train Acc: {100. * correct / total:.2f}%, Loss: {running_loss / len(train_loader):.3f}")


def test(model, test_loader, device, epoch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"[Epoch {epoch}] Test  Acc: {100. * correct / total:.2f}%")
    return 100. * correct / total