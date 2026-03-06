"""
Standalone Patient Simulation API

A self-contained FastAPI service that provides real-time voice-based patient
simulation using FastRTC (WebRTC/WebSocket) with Whisper STT, OpenRouter LLM,
and Inworld TTS.

No auth, no database. Sessions are identified by server-generated UUIDs and
held in memory. The conversation transcript is returned when the session ends.

Usage:
    uvicorn simulation_api.main:app --reload

Endpoints:
    POST /sessions                    Create and start a session
    POST /sessions/{id}/stream/webrtc/offer    WebRTC handshake (auto-mounted)
    WS   /sessions/{id}/stream/websocket/offer WebSocket stream (auto-mounted)
    POST /sessions/{id}/end           End session, retrieve transcript
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simulation_api.config import settings
from simulation_api.routes.sessions import router as sessions_router

app = FastAPI(
    title="Patient Simulation API",
    description="Standalone real-time voice patient simulation service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)
