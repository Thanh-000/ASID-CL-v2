"""
Multimodal Head Architectures for Continual Model Merging.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearExpert(nn.Module):
    """Baseline Unimodal / Naive Linear Expert."""
    def __init__(self, in_features: int = 512, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(F.normalize(x, dim=-1)))
        return self.classifier(h)


class UnimodalExpert(nn.Module):
    """Expert chuyên trách đơn phương thức (Vision-Only hoặc Text-Only)."""
    def __init__(self, in_features: int = 512, hidden_dim: int = 256, num_classes: int = 3, modality: str = "vision"):
        super().__init__()
        if modality not in ("vision", "text"):
            raise ValueError(f"Unsupported modality '{modality}'. Expected 'vision' or 'text'.")
        self.modality = modality
        self.proj = nn.Linear(in_features, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=True)

    def forward(self, img: Optional[torch.Tensor] = None, txt: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = img if self.modality == "vision" else txt
        if feat is None:
            raise ValueError(f"Feature for modality '{self.modality}' cannot be None.")
        h = F.relu(self.proj(F.normalize(feat, dim=-1)))
        return self.classifier(h)


class DualStreamCrossGatedExpert(nn.Module):
    """
    Multimodal Dual-Stream Cross-Gated Expert Head.
    
    Đặc tính thiết kế độc quyền của MM-FLEX-MERGE:
    1. Chiếu song song đặc trưng Thị giác (Image) và Ngôn ngữ (Text) vào không gian tiềm ẩn chung.
    2. Cổng điều hòa tương hỗ (Cross-Modal Attention Gate) sinh trọng số động alpha_img, alpha_txt.
    3. Tự thích ứng kháng suy hao (Missing-Modality Invariant): Tự động phát hiện khi 1 kênh bị mất
       hoặc che mờ hoàn toàn để dồn 100% trọng số sang kênh còn lại.
    """
    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.temperature = temperature

        # Nhánh chiếu tiềm ẩn song song
        self.proj_img = nn.Linear(in_features, hidden_dim, bias=False)
        self.proj_txt = nn.Linear(in_features, hidden_dim, bias=False)

        # Cổng điều hòa tương hỗ (Cross-Modal Gate)
        self.gate_network = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2, bias=True),
        )

        # Đầu phân lớp hợp nhất
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=True)

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        # Chuẩn hóa L2 vector nhúng đầu vào
        img_norm = torch.norm(img, dim=-1, keepdim=True) + 1e-8
        txt_norm = torch.norm(txt, dim=-1, keepdim=True) + 1e-8

        img_n = img / img_norm
        txt_n = txt / txt_norm

        # Kiểm tra sự tồn tại của phương thức (kháng khuyết thiếu)
        img_is_active = (img_norm > 1e-5).float()
        txt_is_active = (txt_norm > 1e-5).float()

        # Tính toán đặc trưng tiềm ẩn
        h_img = self.proj_img(img_n)
        h_txt = self.proj_txt(txt_n)

        # Trọng số cổng điều hòa
        gate_input = torch.cat([img_n, txt_n], dim=-1)
        gate_logits = self.gate_network(gate_input) / self.temperature
        raw_weights = F.softmax(gate_logits, dim=-1) # [B, 2]

        # Hiệu chỉnh thích ứng khi 1 kênh bị suy hao
        w_img = raw_weights[:, 0:1] * img_is_active
        w_txt = raw_weights[:, 1:2] * txt_is_active
        sum_w = w_img + w_txt + 1e-8
        w_img = w_img / sum_w
        w_txt = w_txt / sum_w

        # Tổng hợp biểu diễn đa phương thức
        h_fused = F.relu(w_img * h_img + w_txt * h_txt)
        return self.classifier(h_fused)

    def extract_flat_parameters(self) -> torch.Tensor:
        """Trích xuất toàn bộ tham số của Head thành 1 vector 1 chiều liên tục."""
        return torch.cat([p.detach().clone().flatten() for p in self.parameters()])

    def load_flat_parameters(self, flat_params: torch.Tensor):
        """Nạp vector tham số phẳng vào các layer của Model."""
        idx = 0
        for p in self.parameters():
            num = p.numel()
            p.data.copy_(flat_params[idx : idx + num].view(p.shape))
            idx += num
