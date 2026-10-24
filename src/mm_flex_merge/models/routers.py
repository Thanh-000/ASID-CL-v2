"""
Dual-Centroid Multimodal Router with Negative Routing Bias and 0-Second Data Opt-Out.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualCentroidRouter(nn.Module):
    """
    Multimodal Dual-Centroid Router for Task / Expert Routing.
    
    Đặc tính thiết kế:
    1. Quản lý đồng thời Centroid Thị giác và Centroid Ngôn ngữ của từng Task.
    2. Tích hợp vector Negative Routing Bias (b_k) để chống thiên lệch phân bố nhiệm vụ.
    3. Tích hợp cơ chế Zero-Latency Machine Unlearning (0-second Data Opt-Out):
       Hạ Logit của Task cần xóa xuống -10^9 để xác suất Softmax tiến về 0.0000% tuyệt đối.
    """
    def __init__(self, num_tasks: int, dim: int = 512, default_bias: float = -0.25):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.default_bias = default_bias

        # Ma trận Centroid độc lập
        self.image_centroids = nn.Parameter(torch.zeros(num_tasks, dim), requires_grad=False)
        self.text_centroids = nn.Parameter(torch.zeros(num_tasks, dim), requires_grad=False)

        # Vector Bias điều hòa
        init_bias = torch.full((num_tasks,), default_bias)
        init_bias[0] = 0.0 # Anchor task giữ bias = 0
        self.biases = nn.Parameter(init_bias, requires_grad=False)
        self.opt_out_mask = nn.Parameter(torch.ones(num_tasks, dtype=torch.bool), requires_grad=False)

    def update_centroids(self, task_idx: int, train_img: torch.Tensor, train_txt: torch.Tensor):
        """Cập nhật tọa độ Centroid tâm cụm dựa trên dữ liệu huấn luyện của Task."""
        c_img = F.normalize(train_img.mean(dim=0), dim=-1)
        c_txt = F.normalize(train_txt.mean(dim=0), dim=-1)
        self.image_centroids.data[task_idx].copy_(c_img)
        self.text_centroids.data[task_idx].copy_(c_txt)

    def get_joint_centroids(self, beta: float = 0.5) -> torch.Tensor:
        """Sinh ma trận Centroid liên hợp đa phương thức."""
        joint = beta * self.image_centroids + (1.0 - beta) * self.text_centroids
        return F.normalize(joint, dim=-1)

    def route(self, query_img: torch.Tensor, query_txt: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        """
        Định tuyến mẫu kiểm thử đến các Task Experts.
        Trả về phân bố xác suất Softmax [Batch_Size, Num_Tasks].
        Tự động chuyển sang tâm cụm đơn phương thức khi một kênh bị khuyết (||x_m||_2 <= 1e-5).
        """
        img_active = (torch.norm(query_img, dim=-1, keepdim=True) > 1e-5)
        txt_active = (torch.norm(query_txt, dim=-1, keepdim=True) > 1e-5)

        q_img = F.normalize(query_img, dim=-1)
        q_txt = F.normalize(query_txt, dim=-1)
        q_joint = F.normalize(beta * q_img + (1.0 - beta) * q_txt, dim=-1)

        joint_centroids = self.get_joint_centroids(beta)
        sim_joint = q_joint @ joint_centroids.T
        sim_img = q_img @ F.normalize(self.image_centroids, dim=-1).T
        sim_txt = q_txt @ F.normalize(self.text_centroids, dim=-1).T

        both_active = img_active & txt_active
        sim = torch.where(
            both_active,
            sim_joint,
            torch.where(img_active, sim_img, sim_txt),
        )
        logits = sim + self.biases.unsqueeze(0)

        # Nếu task bị opt-out, dìm logit xuống cực âm
        logits = torch.where(self.opt_out_mask.unsqueeze(0), logits, torch.tensor(-1e9, device=logits.device))
        return F.softmax(logits, dim=-1)

    def opt_out_task(self, task_idx: int):
        """
        Cơ chế 0-Second Data Opt-Out:
        Gỡ bỏ hoàn toàn Task khỏi hệ thống suy luận với thời gian trễ 0.000 giây.
        """
        self.biases.data[task_idx] = -1e9
        self.opt_out_mask.data[task_idx] = False

    def restore_task(self, task_idx: int):
        """Khôi phục lại Task nếu có yêu cầu tái kích hoạt."""
        self.biases.data[task_idx] = 0.0 if task_idx == 0 else self.default_bias
        self.opt_out_mask.data[task_idx] = True

    def is_opted_out(self, task_idx: int) -> bool:
        return not bool(self.opt_out_mask[task_idx].item())
