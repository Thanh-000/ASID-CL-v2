"""
Task Expert Trainer with Reproducibility Provenance.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from ..models.heads import DualStreamCrossGatedExpert


class TaskExpertTrainer:
    """Bộ huấn luyện chuyên gia đa phương thức cho từng Task tuần tự."""
    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 3,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        epochs: int = 30,
    ):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def train(
        self,
        train_img: torch.Tensor,
        train_txt: torch.Tensor,
        train_y: torch.Tensor,
        seed: int = 42,
    ) -> DualStreamCrossGatedExpert:
        torch.manual_seed(seed)
        model = DualStreamCrossGatedExpert(
            in_features=self.in_features,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = model(train_img, train_txt)
            loss = loss_fn(logits, train_y)
            loss.backward()
            optimizer.step()

        model.eval()
        return model
