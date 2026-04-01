from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from src.models import GNNEncoder, LinkPredictor


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


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_device(data, device: torch.device):
    data = data.clone()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_label_index = data.edge_label_index.to(device)
    data.edge_label = data.edge_label.to(device)
    return data


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

    train_data = _to_device(train_data, device)
    val_data = _to_device(val_data, device)
    test_data = _to_device(test_data, device)

    model = LinkPredictor(
        GNNEncoder(
            model_name=config.model_name,
            in_channels=train_data.x.size(-1),
            hidden_channels=config.hidden_dim,
            out_channels=config.out_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            gat_heads=config.gat_heads,
        )
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = []
    train_times = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        start = time.perf_counter()

        optimizer.zero_grad()
        logits = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            train_data.edge_label.float(),
        )
        loss.backward()
        optimizer.step()

        epoch_time = time.perf_counter() - start
        train_times.append(epoch_time)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            val_metrics = evaluate(model, val_data)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "val_auc": val_metrics["auc"],
                    "val_ap": val_metrics["ap"],
                }
            )

    val_metrics = evaluate(model, val_data)
    test_metrics = evaluate(model, test_data)

    inference_start = time.perf_counter()
    _ = evaluate(model, test_data)
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
