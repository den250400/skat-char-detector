import torch
import numpy as np

from model import Model
from dataloader import create_dataloaders_from_numpy
from procedures import test


MODEL_PATH = "models/0.pth"

images = np.load("./data/images.npy")
labels = np.load("./data/labels.npy")

train_loader, val_loader = create_dataloaders_from_numpy(images=np.load("./data/images.npy"),
                                                         labels=np.load("./data/labels.npy"))
model = Model(n_classes=35, leaky_slope=0.1)
model.load_state_dict(torch.load(MODEL_PATH))


test(model=model,
     test_loader=val_loader)
