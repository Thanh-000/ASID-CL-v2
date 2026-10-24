"""
Structured Experiment Logging Utilities.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    Logger cấu trúc cho thí nghiệm khoa học.

    Ghi nhận hyperparams, metrics, provenance vào file JSON với timestamp.
    Mỗi experiment run tạo một file riêng biệt trong output_dir.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "analysis_outputs",
        run_id: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_data: Dict[str, Any] = {
            "experiment": experiment_name,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": {},
            "metrics": {},
            "provenance": {},
        }

    def log_config(self, config: Dict[str, Any]) -> None:
        """Ghi nhận cấu hình hyperparameters."""
        self.log_data["config"].update(config)

    def log_metrics(self, phase: str, metrics: Dict[str, Any]) -> None:
        """Ghi nhận metrics của một pha thí nghiệm."""
        if phase not in self.log_data["metrics"]:
            self.log_data["metrics"][phase] = {}
        self.log_data["metrics"][phase].update(metrics)

    def log_provenance(self, info: Dict[str, Any]) -> None:
        """Ghi nhận thông tin xuất xứ (seed, hardware, versions)."""
        self.log_data["provenance"].update(info)

    def save(self) -> Path:
        """Lưu toàn bộ log ra file JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.experiment_name}_{self.run_id}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        return filepath
