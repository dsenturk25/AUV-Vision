import torch
from torch import nn
from pathlib import Path


def save_model(model: nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"

    model_save_path = target_dir_path / model_name

    print(f"[INFO] saving model at {model_save_path}...")
    torch.save(obj=model.state_dict(), f=model_save_path)


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100

    return acc


def print_train_time(start: float, end: float, device: torch.device):
    total_time = abs(start - end)
    print(f"Train time on {device}: {total_time:.3f}")
