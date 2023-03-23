import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset, n_classes):
        self.dataset = _dataset
        self.n_classes = n_classes

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return torch.tensor(example, dtype=torch.float).unsqueeze(0), \
               F.one_hot(torch.tensor(target, dtype=torch.int64), self.n_classes).type(torch.float32)

    def __len__(self):
        return len(self.dataset)


def load_data(negative_path: str, positive_path: str, img_size=(48, 48)):
    # Load negative samples (not marker)
    negative_files = [os.path.join(negative_path, filename) for filename in os.listdir(negative_path)]
    negative_images = [cv2.resize(cv2.imread(f, flags=cv2.IMREAD_GRAYSCALE), img_size) for f in negative_files]
    negative_labels = list(np.zeros(len(negative_images)))

    # Load positive samples
    positive_images = []
    positive_labels = []

    dirnames = os.listdir(positive_path)
    dirnames.sort(key=int)
    positive_dirs = [os.path.join(positive_path, dirname) for dirname in dirnames]
    for i in tqdm(range(len(positive_dirs))):
        files = [os.path.join(positive_dirs[i], f) for f in os.listdir(positive_dirs[i])]
        images = [cv2.resize(cv2.imread(f, flags=cv2.IMREAD_GRAYSCALE), img_size) for f in files]
        labels = list(np.ones(len(images)) + i)

        positive_images.extend(images)
        positive_labels.extend(labels)

    images = []
    images.extend(negative_images)
    images.extend(positive_images)
    labels = []
    labels.extend(negative_labels)
    labels.extend(positive_labels)

    return images, labels


def create_dataloaders(negative_path: str, positive_path: str, shuffle: bool = True, batch_size=32, n_classes=35,
                       validation_fraction=0.1):
    images, labels = load_data(negative_path, positive_path)
    data = list(zip(images, labels))
    dataset = CustomDataset(data, n_classes=n_classes)

    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - validation_fraction, validation_fraction],
                                                               generator=gen)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)

    return train_loader, val_loader


def create_dataloaders_from_numpy(images: np.array, labels: np.array, shuffle: bool = True, batch_size=32,
                                  n_classes=35, validation_fraction=0.1):
    data = list(zip(list(images), list(labels)))
    dataset = CustomDataset(data, n_classes=n_classes)

    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - validation_fraction, validation_fraction],
                                                               generator=gen)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)

    return train_loader, val_loader


if __name__ == "__main__":
    images, labels = load_data(negative_path='./data/negative_samples', positive_path='./data/positive_samples')
    np.save("./data/images.npy", np.array(images))
    np.save("./data/labels.npy", np.array(labels))
    create_dataloaders_from_numpy(np.array(images), np.array(labels))
