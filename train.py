import torch.optim as optim

from model import Model
from dataloader import create_dataloaders
from procedures import train, test


SAVE_PATH = "./models"

train_loader, val_loader = create_dataloaders(negative_path='./data/negative_samples',
                                              positive_path='./data/positive_samples')
model = Model(n_classes=37, leaky_slope=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model=model,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=optimizer,
      epochs=1,
      save_path=SAVE_PATH)
