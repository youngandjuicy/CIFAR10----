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


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    device
):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)

    return epoch, best_acc
