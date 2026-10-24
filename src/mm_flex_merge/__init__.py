"""
MM-FLEX-MERGE: Multimodal Dynamic Orthogonal Elastic Merging Framework
======================================================================
Một bộ khung nghiên cứu khoa học chuyên sâu và baseline độc quyền phục vụ
Continual Model Merging, Đa phương thức (Multimodal VLM - CLIP), Lọc không gian con
trực giao (Orthogonal Null-Space) và cơ chế Zero-Latency Compliance (GDPR Data Opt-Out).
"""

# --- Models ---
from .models.analytic import BlockAnalyticCL
from .models.heads import DualStreamCrossGatedExpert, LinearExpert, UnimodalExpert
from .models.nullspace_lora import TEXT_MASK, AttnState, SessionLoRA
from .models.routers import DualCentroidRouter

# --- Merging Operators ---
from .merging.flex_merge import MMFlexMerger
from .merging.opcm import OPCMMerger
from .merging.nufilt import NUFILTMerger
from .merging.arithmetic import TaskArithmeticMerger, SWAMerger

# --- Evaluation Protocols ---
from .protocols.metrics import ContinualMergingProtocol
from .protocols.missing_modality import MissingModalityProtocol
from .protocols.opt_out import DataOptOutProtocol
from .protocols.statistical import StatisticalSignificanceProtocol

# --- Engine ---
from .engine.trainer import TaskExpertTrainer
from .engine.evaluator import StreamEvaluator

# --- Datasets ---
from .datasets.base import BaseEmbeddingDataset
from .datasets.uk_crisismmd import UKCrisisMMDDataset
from .datasets.crisismmd_v2 import CrisisMMDv2Dataset

# --- Utilities ---
from .utils.seed import set_deterministic_seed
from .utils.io import load_npz_cache, save_npz_cache

__version__ = "1.1.0"
__all__ = [
    # Models
    "AttnState",
    "BlockAnalyticCL",
    "DualStreamCrossGatedExpert",
    "LinearExpert",
    "SessionLoRA",
    "TEXT_MASK",
    "UnimodalExpert",
    "DualCentroidRouter",
    # Merging Operators
    "MMFlexMerger",
    "OPCMMerger",
    "NUFILTMerger",
    "TaskArithmeticMerger",
    "SWAMerger",
    # Protocols
    "ContinualMergingProtocol",
    "MissingModalityProtocol",
    "DataOptOutProtocol",
    "StatisticalSignificanceProtocol",
    # Engine
    "TaskExpertTrainer",
    "StreamEvaluator",
    # Datasets
    "BaseEmbeddingDataset",
    "UKCrisisMMDDataset",
    "CrisisMMDv2Dataset",
    # Utilities
    "set_deterministic_seed",
    "load_npz_cache",
    "save_npz_cache",
]
