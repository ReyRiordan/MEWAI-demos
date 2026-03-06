"""
Pydantic schemas for the standalone patient simulation API.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SpeechConfig(BaseModel):
    provider: str   # "inworld" or "kokoro"
    model: str      # e.g. "inworld-tts-1"
    voice: str      # e.g. "Craig"


class Demographics(BaseModel):
    name: str
    date_of_birth: str   # YYYY-MM-DD
    sex: str             # "male", "female", "intersex"
    gender: str          # "man", "woman", "non-binary"
    background: str


class PatientCase(BaseModel):
    demographics: Demographics
    chief_concern: str
    free_information: List[str]
    locked_information: List[str]
    behavior: Optional[str] = None


class CreateSessionRequest(BaseModel):
    speech: SpeechConfig
    case: PatientCase
    time_limit: Optional[int] = 300  # seconds


class TranscriptMessage(BaseModel):
    role: str       # "student" or "patient"
    content: str
    timestamp: datetime


class CreateSessionResponse(BaseModel):
    session_id: str
    stream_url: str
    expires_in: int


class EndSessionResponse(BaseModel):
    session_id: str
    transcript: List[TranscriptMessage]
    duration_seconds: int
