# utils/checkpoint.py

import torch

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_acc,
    path
):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc
        },
        path
    )