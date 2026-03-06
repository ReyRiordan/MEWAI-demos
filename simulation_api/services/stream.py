"""
FastRTC stream management for the standalone patient simulation API.

Handles session creation, in-memory state, and teardown.
No database — transcript is returned on session end.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple

import numpy as np
from fastrtc import AdditionalOutputs, AlgoOptions, ReplyOnPause, Stream, get_twilio_turn_credentials
from numpy.typing import NDArray

from simulation_api.config import settings
from simulation_api.schemas import PatientCase, SpeechConfig, TranscriptMessage
from simulation_api.services.ai import OpenRouterChat, get_stt_model, get_tts_model
from simulation_api.services.prompt_builder import build_patient_prompt

logger = logging.getLogger(__name__)

# In-memory registry of active sessions
# Key: session_id, Value: SimSession
active_sessions: Dict[str, "SimSession"] = {}


@dataclass
class SimSession:
    """Active simulation session state."""

    session_id: str
    speech: SpeechConfig
    system_prompt: str
    transcript: List[TranscriptMessage] = field(default_factory=list)
    stream: Optional[Stream] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    time_limit: int = 300


def create_stream_handler(session: "SimSession"):
    """
    Create FastRTC handler for a simulation session.

    Processes: Audio → STT → LLM → TTS
    """

    async def handler(
        audio: Tuple[int, NDArray[np.int16 | np.float32]],
        session_id: str,
    ) -> AsyncIterator:
        try:
            stt = get_stt_model("fireworks")
            text = await stt.transcribe(audio)
            if not text.strip():
                return

            logger.info(f"[{session.session_id}] Student: {text}")

            student_msg = TranscriptMessage(
                role="student", content=text, timestamp=datetime.utcnow()
            )
            session.transcript.append(student_msg)

            yield AdditionalOutputs(
                [{"role": m.role, "content": m.content} for m in session.transcript]
            )

            llm = OpenRouterChat(settings.openrouter_api_key, settings.openrouter_base_url)

            messages = [
                {
                    "role": "user" if m.role == "student" else "assistant",
                    "content": m.content,
                }
                for m in session.transcript
            ]

            response_text = await llm.chat(messages, session.system_prompt)

            logger.info(f"[{session.session_id}] Patient: {response_text}")

            patient_msg = TranscriptMessage(
                role="patient", content=response_text, timestamp=datetime.utcnow()
            )
            session.transcript.append(patient_msg)

            tts = get_tts_model(
                provider=session.speech.provider,
                voice=session.speech.voice,
                model=session.speech.model,
            )
            for audio_chunk in tts.stream_tts_sync(response_text):
                yield audio_chunk

            yield AdditionalOutputs(
                [{"role": m.role, "content": m.content} for m in session.transcript]
            )

        except Exception as e:
            logger.error(f"[{session.session_id}] Stream handler error: {e}", exc_info=True)

            error_msg = "I'm having trouble right now. Could you please repeat that?"
            try:
                tts = get_tts_model(
                    provider=session.speech.provider,
                    voice=session.speech.voice,
                    model=session.speech.model,
                )
                for audio_chunk in tts.stream_tts_sync(error_msg):
                    yield audio_chunk
            except Exception as tts_error:
                logger.error(f"[{session.session_id}] TTS error during error handling: {tts_error}")

    return handler


def validate_voice_config(speech: SpeechConfig) -> None:
    voices_path = Path(__file__).parent.parent / "resources" / "voices.json"
    voices = json.loads(voices_path.read_text())

    for voice in voices:
        if (
            voice["provider"] == speech.provider
            and voice["model"] == speech.model
            and voice["voice"] == speech.voice
        ):
            return

    raise ValueError(
        f"Voice config not found in registry: {speech.provider}/{speech.model}/{speech.voice}"
    )


def create_session(session_id: str, case: PatientCase, speech: SpeechConfig, time_limit: int, app) -> SimSession:
    """
    Create a SimSession, mount a FastRTC stream, and register it.

    Args:
        session_id: UUID for the session
        case: Patient case details
        speech: TTS voice configuration
        time_limit: Session duration in seconds
        app: FastAPI application instance

    Returns:
        The created SimSession
    """
    validate_voice_config(speech)

    system_prompt = build_patient_prompt(case)

    session = SimSession(
        session_id=session_id,
        speech=speech,
        system_prompt=system_prompt,
        time_limit=time_limit,
    )

    handler = create_stream_handler(session)

    rtc_config = None
    if settings.twilio_account_sid and settings.twilio_auth_token:
        rtc_config = get_twilio_turn_credentials(
            account_sid=settings.twilio_account_sid,
            auth_token=settings.twilio_auth_token,
        )

    algo_options = AlgoOptions(
        audio_chunk_duration=0.6,
        started_talking_threshold=0.3,
        speech_threshold=0.3,
    )

    stream = Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(handler, input_sample_rate=16000, algo_options=algo_options),
        rtc_configuration=rtc_config,
        concurrency_limit=settings.concurrency_limit,
        time_limit=time_limit,
    )

    session.stream = stream

    mount_path = f"/sessions/{session_id}/stream"
    stream.mount(app, path=mount_path)

    active_sessions[session_id] = session
    logger.info(f"Mounted stream for session {session_id} at {mount_path}")

    return session


def end_session(session_id: str) -> Optional[SimSession]:
    """
    Close a session's stream and remove it from the registry.

    Returns the session (with transcript) or None if not found.
    """
    session = active_sessions.pop(session_id, None)
    if not session:
        return None

    if session.stream and hasattr(session.stream, "close"):
        try:
            session.stream.close()
        except Exception as e:
            logger.warning(f"Error closing stream for session {session_id}: {e}")

    logger.info(f"Ended session {session_id} with {len(session.transcript)} messages")
    return session
