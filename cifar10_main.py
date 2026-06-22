import os
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from configs.config import Config
from datasets.cifar10 import get_dataloader
from models.tudui import Tudui
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from utils.checkpoint import save_checkpoint
from utils.seed import set_seed

cfg = Config()

set_seed(cfg.seed)

train_loader, test_loader = get_dataloader(cfg)

model = Tudui(num_classes=cfg.num_classes)
model.to(cfg.device)

criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

os.makedirs(cfg.checkpoint_dir, exist_ok=True)
best_acc = 0.0

for epoch in range(cfg.num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=cfg.device, epoch=epoch+1)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device=cfg.device)

    print(f"Epoch [{epoch+1}/{cfg.num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

    if cfg.save_best and test_acc > best_acc:
        best_acc = test_acc
        save_path = os.path.join(cfg.checkpoint_dir, "best.pth")
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_acc, save_path)
        print(f"Saved best checkpoint: epoch {epoch+1}, best acc {best_acc*100:.2f}%")

    scheduler.step()
