"""
Unified Multi-Task Stream Evaluator and Matrix Telemetry.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from ..protocols.metrics import ContinualMergingProtocol


class StreamEvaluator:
    """Bộ đánh giá toàn diện trên toàn bộ chuỗi Task Stream."""
    @staticmethod
    def evaluate_expert(
        expert,
        test_img: torch.Tensor,
        test_txt: torch.Tensor,
        test_y: torch.Tensor,
    ) -> Tuple[float, float]:
        expert.eval()
        with torch.no_grad():
            logits = expert(test_img, test_txt)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = test_y.cpu().numpy()
        return ContinualMergingProtocol.evaluate_predictions(targets, preds)

    @staticmethod
    def evaluate_routed_ensemble(
        router,
        experts: List,
        test_img: torch.Tensor,
        test_txt: torch.Tensor,
        test_y: torch.Tensor,
    ) -> Tuple[float, float]:
        with torch.no_grad():
            route_weights = router.route(test_img, test_txt) # [B, Num_Tasks]
            expert_logits = torch.stack([exp(test_img, test_txt) for exp in experts], dim=1) # [B, Num_Tasks, Num_Classes]
            final_logits = (route_weights.unsqueeze(-1) * expert_logits).sum(dim=1) # [B, Num_Classes]
            preds = torch.argmax(final_logits, dim=-1).cpu().numpy()
            targets = test_y.cpu().numpy()
        return ContinualMergingProtocol.evaluate_predictions(targets, preds)
