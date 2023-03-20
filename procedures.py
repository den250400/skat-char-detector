import torch
import os
import torch.nn as nn
from tqdm import tqdm


def train(model, train_loader, val_loader, optimizer, epochs, save_path, validation=True, save_period=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        if validation:
            test(model, val_loader)

        if epoch % save_period == 0:
            model.to('cpu')
            torch.save(model.state_dict(), os.path.join(save_path, "%i.pth" % epoch))


def test(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).sum().item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.argmax(dim=1).view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nValidation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
