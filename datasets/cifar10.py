# CIFAR10数据加载器

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(cfg):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.ToTensor()

    train_set = CIFAR10(
        root=cfg.data_root,
        train=True,
        transform=train_transform,
        download=True
    )

    test_set = CIFAR10(
        root=cfg.data_root,
        train=False,
        transform=test_transform,
        download=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False
    )

    return train_loader, test_loader