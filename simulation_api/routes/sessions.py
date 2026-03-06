"""
Session endpoints for the standalone patient simulation API.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from simulation_api.schemas import CreateSessionRequest, CreateSessionResponse, EndSessionResponse
from simulation_api.services.stream import active_sessions, create_session, end_session

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse, status_code=201)
async def start_session(body: CreateSessionRequest, request: Request):
    """
    Create and start a new patient simulation session.

    Returns a session_id and the stream URL to connect to via WebRTC or WebSocket.
    """
    session_id = str(uuid.uuid4())

    try:
        create_session(
            session_id=session_id,
            case=body.case,
            speech=body.speech,
            time_limit=body.time_limit,
            app=request.app,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return CreateSessionResponse(
        session_id=session_id,
        stream_url=f"/sessions/{session_id}/stream",
        expires_in=body.time_limit,
    )


@router.post("/{session_id}/end", response_model=EndSessionResponse)
async def stop_session(session_id: str):
    """
    End a simulation session and retrieve the full conversation transcript.

    Can be called explicitly by the client or after the session has auto-timed out.
    If the session has already been ended, returns 404.
    """
    session = end_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or already ended")

    elapsed = int((datetime.utcnow() - session.created_at).total_seconds())

    return EndSessionResponse(
        session_id=session_id,
        transcript=session.transcript,
        duration_seconds=elapsed,
    )
