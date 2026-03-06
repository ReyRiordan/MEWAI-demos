"""
Speech-to-text services for patient simulation.

Provides async STT implementations for transcribing student audio.
"""

import tempfile
from abc import ABC, abstractmethod
from typing import Tuple

import aiohttp
import numpy as np
import soundfile as sf
from numpy.typing import NDArray


class BaseSTT(ABC):
    """Abstract base class for speech-to-text services."""

    @abstractmethod
    async def transcribe(self, audio: Tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        pass


class WhisperSTT(BaseSTT):
    """Whisper STT via Fireworks API."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.url = base_url

    async def transcribe(self, audio: Tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sr, arr = audio

        if arr.ndim > 1:
            arr = np.squeeze(arr, axis=0)

        if arr.dtype != np.int16:
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, arr, sr, subtype="PCM_16")

        try:
            async with aiohttp.ClientSession() as session:
                with open(temp_path, "rb") as audio_file:
                    form_data = aiohttp.FormData()
                    form_data.add_field("file", audio_file, filename="audio.wav")
                    form_data.add_field("model", "whisper-v3")
                    form_data.add_field("temperature", "0")
                    form_data.add_field("vad_model", "silero")

                    headers = {"Authorization": f"Bearer {self.api_key}"}

                    async with session.post(
                        self.url, headers=headers, data=form_data
                    ) as response:
                        if response.status == 200:
                            output = await response.json()
                            return output.get("text", "")
                        else:
                            error_text = await response.text()
                            raise Exception(
                                f"Transcription failed: {response.status} - {error_text}"
                            )
        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def get_stt_model(provider: str = "fireworks") -> BaseSTT:
    from simulation_api.config import settings

    if provider == "fireworks":
        return WhisperSTT(settings.fireworks_api_key, settings.fireworks_base_url)
    else:
        raise ValueError(f"Unknown STT provider: {provider}")
