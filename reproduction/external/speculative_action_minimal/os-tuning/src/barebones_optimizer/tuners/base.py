from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TunerResponse:
    proposed_parameters: Dict[str, Any]
    justification: str
    raw_text: str
    token_metrics: Optional[Dict[str, int]]
