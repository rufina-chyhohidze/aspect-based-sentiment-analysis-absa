from dataclasses import dataclass
from typing import List


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str
    confidence: float
    text_span: List[int]


class ABSAAnalyzer:
    """Base class/interface for all ABSA implementations"""
    def analyze(self, text: str) -> List[AspectSentiment]:
        raise NotImplementedError
