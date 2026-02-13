from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class Q1ModelMetrics:
    dev_raw_em: float
    dev_norm_em: float
    test_raw_em: float
    test_norm_em: float
    training_time_s: float


@dataclass
class Q1MetricsPayload:
    generated_at: str
    source: str
    models: Dict[str, Q1ModelMetrics]
    errors: Dict[str, Dict[str, int]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Q2MetricRow:
    base: float
    fine_tuned: float
    improvement_pct: float


@dataclass
class Q2MetricsPayload:
    generated_at: str
    source: str
    metrics: Dict[str, Q2MetricRow]

    def to_dict(self) -> dict:
        return asdict(self)
