"""
Text-to-speech services for patient simulation.

Provides TTS implementations for generating patient voice responses.
"""

import base64
import json
from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import numpy as np
import requests
from fastrtc import KokoroTTSOptions, get_tts_model as fastrtc_get_tts_model
from numpy.typing import NDArray


class BaseTTS(ABC):
    """Abstract base class for text-to-speech services."""

    @abstractmethod
    def stream_tts_sync(self, text: str) -> Iterator[Tuple[int, NDArray]]:
        pass


class InworldTTS(BaseTTS):
    """Inworld TTS implementation. Streams LINEAR16 audio at 48kHz."""

    def __init__(self, api_key: str, voice: str, speed: float = 1.0):
        self.api_key = api_key
        self.voice = voice
        self.speed = speed
        self.url = "https://api.inworld.ai/tts/v1/voice:stream"

    def stream_tts_sync(self, text: str) -> Iterator[Tuple[int, NDArray]]:
        payload = {
            "text": text,
            "voiceId": self.voice,
            "modelId": "inworld-tts-1",
            "audio_config": {
                "audio_encoding": "LINEAR16",
                "sample_rate_hertz": 48000,
                "speakingRate": self.speed,
            },
        }

        headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.url, json=payload, headers=headers, stream=True)
        response.raise_for_status()

        sample_rate = payload["audio_config"]["sample_rate_hertz"]

        for line in response.iter_lines():
            if not line:
                continue
            try:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                chunk = json.loads(line)

                audio_chunk = base64.b64decode(chunk["result"]["audioContent"])

                if len(audio_chunk) > 44:
                    pcm = audio_chunk[44:]
                    waveform = (
                        np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    yield (sample_rate, waveform)

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}, Line content: {line}")
                continue
            except Exception as e:
                print(f"Error processing chunk: {e}, Line: {line}")
                continue


class KokoroTTS(BaseTTS):
    """Kokoro TTS implementation using FastRTC."""

    def __init__(self, voice: str, speed: float = 1.2, lang: str = "en-us"):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.options = KokoroTTSOptions(voice=voice, speed=speed, lang=lang)
        self.model = fastrtc_get_tts_model(model="kokoro")

    def stream_tts_sync(self, text: str) -> Iterator[Tuple[int, NDArray]]:
        if hasattr(self.model, "stream_tts_sync"):
            yield from self.model.stream_tts_sync(text, self.options)
        else:
            raise NotImplementedError(
                "Kokoro TTS streaming requires FastRTC Stream context"
            )


def get_tts_model(provider: str, voice: str, model: str = None, **kwargs) -> BaseTTS:
    from simulation_api.config import settings

    if provider == "inworld":
        if not settings.inworld_api_key:
            raise ValueError("Inworld API key not configured")
        return InworldTTS(
            api_key=settings.inworld_api_key,
            voice=voice,
            speed=kwargs.get("speed", 1.0),
        )
    elif provider == "kokoro":
        return KokoroTTS(
            voice=voice, speed=kwargs.get("speed", 1.2), lang=kwargs.get("lang", "en-us")
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")
