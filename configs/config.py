# 超参数设置
import torch

class Config:
    data_root = "./dataset"
    log_dir = "./CIFAR10_train"
    checkpoint_dir = "./checkpoints"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 64
    num_epochs = 100
    num_classes = 10

    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    num_workers = 0
    seed = 42

    save_best = True
    save_latest = True
    resume = False
    resume_path = "./checkpoints/latest.pth"
