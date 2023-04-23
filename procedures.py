import torch
import os
import torch.nn as nn
from tqdm import tqdm


def detection_loss(output: torch.Tensor, target: torch.Tensor, clf_w: float = 0.5, conf_w: float = 0.5):
    loss_classification = nn.CrossEntropyLoss()
    loss_confidence = nn.BCEWithLogitsLoss()

    loss_clf = torch.mean(target[:, 0:1] * loss_classification(output[:, 1:], target[:, 1:]))
    loss_conf = loss_confidence(output[:, 0], target[:, 0])
    loss = clf_w * loss_clf + conf_w * loss_conf

    return loss


def train(model, train_loader, val_loader, optimizer, epochs, save_path, validation=True, save_period=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for epoch in range(epochs):
        model.train()
        model.to(device)
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = detection_loss(output, target)
            loss.backward()
            optimizer.step()

        if epoch % save_period == 0:
            model.to('cpu')
            torch.save(model.state_dict(), os.path.join(save_path, "%i.pth" % epoch))

        if validation:
            test(model, val_loader)


def test(model, test_loader, confidence_thresh: float = 0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    test_loss = 0
    correct_clf = 0
    correct_conf = 0
    model.to(device)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += detection_loss(output, target).sum().item()  # sum up batch loss

            # Correct class
            output_filtered = output[target[:, 0] == 1]
            target_filtered = target[target[:, 0] == 1]
            pred = output_filtered[:, 1:].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_clf += pred.eq(target_filtered[:, 1:].argmax(dim=1).view_as(pred)).sum().item()

            # Correct confidence
            pred = (output[:, 0] > confidence_thresh).type(torch.int64)
            correct_conf += pred.eq(target[:, 0]).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nValidation:\nAverage loss: {:.4f}\nMulti-class accuracy: {}/{} ({:.3f}%)\nConfidence accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct_clf, len(test_loader.dataset),
        100. * correct_clf / len(test_loader.dataset),
        correct_conf, len(test_loader.dataset), 100. * correct_conf / len(test_loader.dataset)))
