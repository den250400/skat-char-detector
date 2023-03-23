import torch.optim as optim
import numpy as np

from model import Model
from dataloader import create_dataloaders_from_numpy
from procedures import train, test


SAVE_PATH = "models"

images = np.load("./data/images.npy")
labels = np.load("./data/labels.npy")

train_loader, val_loader = create_dataloaders_from_numpy(images=np.load("./data/images.npy"),
                                                         labels=np.load("./data/labels.npy"))
model = Model(n_classes=35, leaky_slope=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model=model,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=optimizer,
      epochs=1,
      save_path=SAVE_PATH)
