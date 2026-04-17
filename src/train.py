from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

from src.models import GNNEncoder, LightGCNEncoder, LinkPredictor


@dataclass
class TrainConfig:
    model_name: str
    hidden_dim: int
    out_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    gat_heads: int = 2
    eval_every: int = 1
    use_amp: bool = True
    use_compile: bool = False
    show_progress: bool = False
    early_stopping_patience: int = 0


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _to_device(data, device: torch.device):
    required = ("x", "edge_index", "edge_label_index", "edge_label")
    if all(
        hasattr(data, name) and getattr(data, name) is not None and getattr(data, name).device == device
        for name in required
    ):
        return data
    return data.to(device)


def _build_model(train_data, config: TrainConfig, device: torch.device) -> LinkPredictor:
    model_name = config.model_name.lower()
    if model_name == "lightgcn":
        encoder = LightGCNEncoder(
            in_channels=train_data.x.size(-1),
            out_channels=config.out_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    else:
        encoder = GNNEncoder(
            model_name=model_name,
            in_channels=train_data.x.size(-1),
            hidden_channels=config.hidden_dim,
            out_channels=config.out_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            gat_heads=config.gat_heads,
        )
    model = LinkPredictor(encoder).to(device)

    if config.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _eval_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_prob >= 0.5).astype(np.int32)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def train_one_model(
    train_data,
    val_data,
    test_data,
    config: TrainConfig,
    seed: int,
    device: torch.device,
):
    _seed_everything(seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_data = _to_device(train_data, device)
    val_data = _to_device(val_data, device)
    test_data = _to_device(test_data, device)

    model = _build_model(train_data, config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    has_torch_amp = hasattr(torch, "amp") and hasattr(torch.amp, "autocast") and hasattr(torch.amp, "GradScaler")
    if has_torch_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history = []
    train_times = []
    best_val_auc = -1.0
    best_state = None
    epochs_since_improvement = 0

    eval_every = max(1, config.eval_every)
    progress = tqdm(
        range(1, config.epochs + 1),
        desc=f"{config.model_name} epochs",
        leave=False,
        disable=not config.show_progress,
    )

    for epoch in progress:
        model.train()
        _sync_if_cuda(device)
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if has_torch_amp:
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)

        with autocast_ctx:
            logits = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                train_data.edge_label.float(),
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        _sync_if_cuda(device)
        epoch_time = time.perf_counter() - start
        train_times.append(epoch_time)

        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == config.epochs
        if should_eval:
            val_metrics = evaluate(model, val_data)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "val_auc": val_metrics["auc"],
                    "val_ap": val_metrics["ap"],
                }
            )
            progress.set_postfix(loss=float(loss.item()), val_auc=val_metrics["auc"])

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if (
                config.early_stopping_patience > 0
                and epochs_since_improvement >= config.early_stopping_patience
            ):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_data)
    test_metrics = evaluate(model, test_data)

    _sync_if_cuda(device)
    inference_start = time.perf_counter()
    _ = evaluate(model, test_data)
    _sync_if_cuda(device)
    inference_time = time.perf_counter() - inference_start

    extras = {
        "avg_train_time_per_epoch": float(np.mean(train_times)),
        "inference_latency_s": float(inference_time),
        "num_parameters": model.num_trainable_parameters(),
        "history": history,
        "model": model,
    }

    return val_metrics, test_metrics, extras


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_label_index)

    y_true = data.edge_label.detach().cpu().numpy().astype(np.int32)
    y_logits = logits.detach().cpu().numpy()
    return _eval_metrics(y_true, y_logits)
