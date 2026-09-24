"""
Schema bất biến — interface giữa mọi module.
Mọi model đều trả Prediction, mọi postprocess đều nhận/trả Segment.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class Window:
    """Cửa sổ thời gian dùng chung cho 3 modality."""
    window_id: int
    start: float          # seconds
    end: float            # seconds


@dataclass
class Prediction:
    """Output chuẩn của 1 model trên 1 window."""
    window_id: int
    start: float
    end: float
    class_id: int
    class_name: str
    confidence: float
    probabilities: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Segment:
    """Đoạn activity liên tục sau post-processing."""
    start: float
    end: float
    class_id: int
    class_name: str
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class FusionRecord:
    """Ghi lại prediction của cả 3 modality + kết quả fusion."""
    window_id: int
    start: float
    end: float
    rgb: Optional[Prediction] = None
    skeleton: Optional[Prediction] = None
    imu: Optional[Prediction] = None
    fusion: Optional[Prediction] = None
