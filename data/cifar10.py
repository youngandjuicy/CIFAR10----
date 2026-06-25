# CIFAR10数据加载器

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(cfg):
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

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
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda")
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda")
    )

    return train_loader, test_loader
