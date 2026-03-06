from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, model_validator


class TranscriptMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class TranscriptRubricItem(BaseModel):
    category: str
    label: str
    description: str
    points: int


class ResponseRubricItem(BaseModel):
    name: str
    description: str
    response: str
    context_responses: List[str] = []
    type: Literal["feature-based", "score-based", "mixed"]
    features: Dict[str, str] = {}
    scoring: Dict[int, str] = {}

    @model_validator(mode='after')
    def validate_type_consistency(self) -> 'ResponseRubricItem':
        if self.type == "feature-based":
            if not self.features:
                raise ValueError("type='feature-based' requires non-empty features dict")
            if self.scoring:
                raise ValueError("type='feature-based' requires empty scoring dict")
        elif self.type == "score-based":
            if not self.scoring:
                raise ValueError("type='score-based' requires non-empty scoring dict")
            if self.features:
                raise ValueError("type='score-based' requires empty features dict")
        elif self.type == "mixed":
            if not self.features or not self.scoring:
                raise ValueError("type='mixed' requires both features and scoring to be non-empty")
        return self


class Rubric(BaseModel):
    name: str
    transcript_items: List[TranscriptRubricItem]
    response_items: List[ResponseRubricItem]


class EvaluateRequest(BaseModel):
    rubric: Rubric
    transcript: List[TranscriptMessage]
    responses: Dict[str, str]


# Output models — exact match to backend Evaluation structure

class TranscriptEvaluation(BaseModel):
    rationale: str
    satisfied: bool


class ResponseEvaluation(BaseModel):
    response: str
    features: Dict[str, Dict[str, Union[str, bool]]] = {}
    scoring: Optional[Dict[str, Union[str, int]]] = None
    feedback: str


class Evaluation(BaseModel):
    transcript: Dict[str, TranscriptEvaluation]
    responses: Dict[str, ResponseEvaluation]
