import os
import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# folder with one subfolder per decade (data/small/1960s, data/small/1970s, ...)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "small")

# settings
IMAGE_SIZE = 128
BATCH_SIZE = 32
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# the rest (15%) is used for testing


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=4):
    # resize all images to the same size and turn them into tensors
    # then normalize them, using the mean and std for ImageNet
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ImageFolder uses the folder names as labels (1960s -> 0, 1970s -> 1, ...)
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # split each decade on its own, so every decade has the same
    # number of images in train, validation and test
    rng = random.Random(42)  # fixed seed gives the same split every time
    train_indices = []
    val_indices = []
    test_indices = []

    for class_index in range(len(dataset.classes)):
        # find all images of this decade and shuffle them
        indices = [i for i, label in enumerate(dataset.targets) if label == class_index]
        rng.shuffle(indices)

        # work out how many images of this decade go into each split
        train_size = int(TRAIN_SPLIT * len(indices))
        val_size = int(VAL_SPLIT * len(indices))

        train_indices += indices[:train_size]
        val_indices += indices[train_size:train_size + val_size]
        test_indices += indices[train_size + val_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    # only the training data needs to be shuffled
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset.classes
