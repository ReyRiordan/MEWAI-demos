import asyncio
import json
import os
import sys
from docx import Document

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation_api"))

from evaluation_api.evaluation import evaluate
from evaluation_api.models import Rubric, TranscriptMessage

import streamlit as st


@st.cache_data
def _load_rubric() -> dict:
    path = os.path.join(os.path.dirname(__file__), "resources", "rubric.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def _load_transcript() -> list:
    path = os.path.join(os.path.dirname(__file__), "resources", "transcript.json")
    with open(path) as f:
        return json.load(f)

st.set_page_config(page_title = "MEWAI",
                   page_icon = "🧑‍⚕️",
                   initial_sidebar_state="collapsed",
                   layout="wide")

if "stage" not in st.session_state:
    st.session_state['stage'] = "MAIN"

def set_stage(stage):
    st.session_state['stage'] = stage

if st.session_state['stage'] == "MAIN":
    st.title("MEWAI")
    st.subheader("Temp onboarding references and demos, code: https://github.com/ReyRiordan/MEWAI-demos")
    st.write("Please contact me at **reyriordan@gmail.com** or **rhr58@scarletmail.rutgers.edu** if you have any questions or issues!")
    st.write("*HuggingFace Spaces is unable to host my simulation demo due to versioning issues, if the link is broken please run it yourself with simulation_demo.py")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Evaluation Service")
        st.write(
            "Evaluates student interview transcripts and written responses "
            "against rubric criteria."
        )
        st.button("Reference", key="eval", on_click=set_stage, args=['EVAL_REF'], use_container_width=True)
        st.button("Demo", key="eval_demo", on_click=set_stage, args=['EVAL_DEMO'], use_container_width=True)

    with col2:
        st.header("Simulation Service")
        st.write(
            "Runs real-time voice conversations between students and AI patients "
            "over WebRTC."
        )
        st.button("Reference", key="sim", on_click=set_stage, args=['SIM_REF'], use_container_width=True)
        st.link_button("Demo", url="https://5dd410f0d6d4d7b25d.gradio.live", use_container_width=True)
    
    with col3:
        st.header("My AWS Hosting Plans/Notes")
        st.write("Hastily written doc with some thoughts that I sent to Dr. Yanamala a while back.")
        with open("resources/AWS_hosting_plans.pdf", "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f.read(),  # raw bytes work too
                file_name="MEWAI_AWS_hosting.pdf",
                mime="application/pdf",
                use_container_width=True
            )


elif st.session_state['stage'] == "EVAL_REF":
    st.title("Evaluation Service — API Reference")
    st.write(
        "REST API that accepts a rubric, interview transcript, and written responses, and returns a structured evaluation."
    )

    st.divider()

    # ── Setup ─────────────────────────────────────────────────────────────────────
    st.header("Setup")

    st.subheader("Running the API")
    st.code(
        "cd demo/evaluation_api\n"
        "pip install -r requirements.txt\n"
        "uvicorn app:app --reload   # runs on http://localhost:8000",
        language="bash",
    )

    st.subheader("Environment Variables")
    st.table(
        {
            "Variable": [
                "OPENROUTER_API_KEY",
                "OPENROUTER_BASE_URL",
                "OPENROUTER_MODEL",
            ],
            "Required": ["Yes", "No", "No"],
            "Default": ["—", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6"],
            "Description": [
                "LLM calls",
                "Request base URL",
                "Evaluator model",
            ],
        }
    )

    st.divider()

    # ── Endpoint ──────────────────────────────────────────────────────────────────
    st.header("Endpoint")

    st.markdown("### `POST /evaluate`")
    st.write(
        "Evaluates a student's transcript and written responses against a rubric. "
        "Request body is `EvaluateRequest`; response is `Evaluation`."
    )
    st.code(
        """\
curl -X POST http://localhost:8000/evaluate \\
  -H "Content-Type: application/json" \\
  -d '{
    "rubric": { ... },
    "transcript": [ ... ],
    "responses": { "differential_diagnosis": "1. ACS\\n2. PE" }
  }'""",
        language="bash",
    )

    st.divider()

    # ── Input Models ──────────────────────────────────────────────────────────────
    st.header("Input Models")

    st.subheader("EvaluateRequest")
    st.code(
        """\
{
  "rubric": Rubric,
  "transcript": TranscriptMessage[],
  "responses": { "response_name": "student text", ... }
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["rubric", "transcript", "responses"],
            "Type": ["Rubric", "TranscriptMessage[]", "dict[str, str]"],
            "Required": ["Yes", "Yes", "Yes"],
            "Description": [
                "Rubric defining evaluation criteria",
                "Ordered list of conversation messages",
                "Map of response name → student's written answer",
            ],
        }
    )

    st.subheader("TranscriptMessage")
    st.code(
        """\
{
  "role": "student" | "patient",
  "content": "What brings you in today?",
  "timestamp": "2025-01-15T10:00:01Z"   // optional
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["role", "content", "timestamp"],
            "Type": ["\"student\" | \"patient\"", "str", "datetime (optional)"],
            "Required": ["Yes", "Yes", "No"],
            "Description": [
                "Speaker of this message",
                "Text of the message",
                "ISO 8601 timestamp",
            ],
        }
    )

    st.subheader("Rubric")
    st.code(
        """\
{
  "name": "Chest Pain Rubric",
  "transcript_items": TranscriptRubricItem[],
  "response_items": ResponseRubricItem[]
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["name", "transcript_items", "response_items"],
            "Type": ["str", "TranscriptRubricItem[]", "ResponseRubricItem[]"],
            "Required": ["Yes", "Yes", "Yes"],
            "Description": [
                "Human-readable rubric name",
                "Criteria evaluated against the interview transcript",
                "Criteria evaluated against written responses",
            ],
        }
    )

    st.subheader("TranscriptRubricItem")
    st.code(
        """\
{
  "category": "History Taking",
  "label": "Asked about onset",
  "description": "Student asked when the pain started",
  "points": 2
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["category", "label", "description", "points"],
            "Type": ["str", "str", "str", "int"],
            "Required": ["Yes", "Yes", "Yes", "Yes"],
            "Description": [
                "Grouping category for the criterion",
                "Short identifier used as the key in evaluation output",
                "What the student must do to satisfy this criterion",
                "Points awarded if satisfied",
            ],
        }
    )

    st.subheader("ResponseRubricItem")
    st.code(
        """\
{
  "name": "differential_diagnosis_eval",
  "description": "Quality of differential diagnosis",
  "response": "differential_diagnosis",
  "context_responses": [],
  "type": "feature-based" | "score-based" | "mixed",
  "features": { "includes_acs": "Does the response mention ACS?" },
  "scoring": { 1: "Poor", 2: "Adequate", 3: "Excellent" }
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["name", "description", "response", "context_responses", "type", "features", "scoring"],
            "Type": ["str", "str", "str", "str[]", "\"feature-based\"|\"score-based\"|\"mixed\"", "dict[str, str]", "dict[int, str]"],
            "Required": ["Yes", "Yes", "Yes", "No", "Yes", "Conditional", "Conditional"],
            "Description": [
                "Unique identifier for this rubric item",
                "What is being evaluated",
                "Key in `responses` dict that this item evaluates",
                "Other response keys to include as context for the LLM",
                "Evaluation strategy (see Rubric Types)",
                "Feature name → description; required for feature-based and mixed",
                "Score → description; required for score-based and mixed",
            ],
        }
    )
    # st.info(
    #     "**Type validation rules:**  \n"
    #     "- `feature-based`: `features` must be non-empty; `scoring` must be empty  \n"
    #     "- `score-based`: `scoring` must be non-empty; `features` must be empty  \n"
    #     "- `mixed`: both `features` and `scoring` must be non-empty"
    # )

    # st.divider()

    # # ── Rubric Types ──────────────────────────────────────────────────────────────
    # st.header("Rubric Types")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("feature-based")
        st.write("Requires `features`, no `scoring`. Boolean check for each feature.")
        st.code(
            """\
{
  "type": "feature-based",
  "features": {
    "mentions_acs": "Does the response list ACS?",
    "mentions_pe": "Does the response list PE?"
  },
  "scoring": {}
}""",
            language="json",
        )

    with col2:
        st.subheader("score-based")
        st.write("Requires `scoring`, no `features`. Assigns a score from the provided scale.")
        st.code(
            """\
{
  "type": "score-based",
  "features": {},
  "scoring": {
    1: "Incomplete",
    2: "Adequate",
    3: "Thorough"
  }
}""",
            language="json",
        )

    with col3:
        st.subheader("mixed")
        st.write("Requires both `features` and `scoring`. Checks each feature and also assigns an overall score, scoring tends to reference which features were satisfied.")
        st.code(
            """\
{
  "type": "mixed",
  "features": {
    "mentions_acs": "Lists ACS?"
  },
  "scoring": {
    1: "No features satisfied.",
    2: "All features satisfied.",
  }
}""",
            language="json",
        )

    st.divider()

    # ── Output Models ─────────────────────────────────────────────────────────────
    st.header("Output Models")

    st.subheader("Evaluation")
    st.code(
        """\
{
  "transcript": { "Asked about onset": TranscriptEvaluation, ... },
  "responses":  { "differential_diagnosis_eval": ResponseEvaluation, ... }
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["transcript", "responses"],
            "Type": ["dict[label → TranscriptEvaluation]", "dict[name → ResponseEvaluation]"],
            "Description": [
                "One entry per TranscriptRubricItem, keyed by `label`",
                "One entry per ResponseRubricItem, keyed by `name`",
            ],
        }
    )

    st.subheader("TranscriptEvaluation")
    st.code(
        """\
{
  "rationale": "Student asked 'When did the pain start?' at turn 3.",
  "satisfied": true
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["rationale", "satisfied"],
            "Type": ["str", "bool"],
            "Description": [
                "LLM explanation of why the criterion was or was not met",
                "Whether the criterion was satisfied",
            ],
        }
    )

    st.subheader("ResponseEvaluation")
    st.code(
        """\
{
  "response": "differential_diagnosis",
  "features": {
    "mentions_acs": { "rationale": "ACS is listed first.", "satisfied": true },
    "mentions_pe":  { "rationale": "PE not mentioned.", "satisfied": false }
  },
  "scoring": { "rationale": "Two of three diagnoses included.", "score": 2 },
  "feedback": ""
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["response", "features", "scoring", "feedback"],
            "Type": [
                "str",
                "dict[feat_name → {rationale: str, satisfied: bool}]",
                "{rationale: str, score: int} | null",
                "str",
            ],
            "Description": [
                "The response key this evaluation corresponds to",
                "Per-feature boolean results (empty for score-based)",
                "Score result (null for feature-based)",
                "Reserved for future use (currently empty string)",
            ],
        }
    )

    st.divider()

    # ── Full Example ──────────────────────────────────────────────────────────────
    st.header("Full Example")

    with st.expander("Request"):
        st.code(
            """\
{
  "rubric": {
    "name": "Chest Pain Rubric",
    "transcript_items": [
      {
        "category": "History Taking",
        "label": "Asked about onset",
        "description": "Student asked when the chest pain started",
        "points": 2
      }
    ],
    "response_items": [
      {
        "name": "differential_diagnosis_eval",
        "description": "Quality of the student's differential diagnosis list",
        "response": "differential_diagnosis",
        "context_responses": [],
        "type": "feature-based",
        "features": {
          "mentions_acs": "Does the response include ACS (acute coronary syndrome)?",
          "mentions_pe":  "Does the response include PE (pulmonary embolism)?"
        },
        "scoring": {}
      }
    ]
  },
  "transcript": [
    { "role": "student", "content": "When did the chest pain start?" },
    { "role": "patient",  "content": "It started about two hours ago." }
  ],
  "responses": {
    "differential_diagnosis": "1. ACS\\n2. PE\\n3. Aortic dissection"
  }
}""",
            language="json",
        )

    with st.expander("Response"):
        st.code(
            """\
{
  "transcript": {
    "Asked about onset": {
      "rationale": "The student directly asked 'When did the chest pain start?' at the start of the interview.",
      "satisfied": true
    }
  },
  "responses": {
    "differential_diagnosis_eval": {
      "response": "differential_diagnosis",
      "features": {
        "mentions_acs": {
          "rationale": "The student lists 'ACS' as their first diagnosis.",
          "satisfied": true
        },
        "mentions_pe": {
          "rationale": "The student lists 'PE' as their second diagnosis.",
          "satisfied": true
        }
      },
      "scoring": null,
      "feedback": ""
    }
  }
}""",
            language="json",
        )

    st.divider()
    st.button("Back", on_click=set_stage, args=["MAIN"], use_container_width=True, type="primary")


elif st.session_state['stage'] == "SIM_REF":
    st.title("Simulation Service — API Reference")
    st.write(
        "REST API that accepts a patient case and time limit and runs a real-time voice "
        "conversation between a student and an AI patient over WebRTC or WebSocket."
    )
    st.divider()

    # ── Setup ─────────────────────────────────────────────────────────────────────
    st.header("Setup")

    st.subheader("Running the API")
    st.code(
        "cd demo/simulation_api\n"
        "pip install -r requirements.txt\n"
        "uvicorn simulation_api.main:app --reload",
        language="bash",
    )

    st.subheader("Environment Variables")
    st.table(
        {
            "Variable": [
                "OPENROUTER_API_KEY",
                "FIREWORKS_API_KEY",
                "INWORLD_API_KEY",
                "TWILIO_ACCOUNT_SID",
                "TWILIO_AUTH_TOKEN",
            ],
            "Required": ["Yes", "Yes", "Yes (if Inworld)", "No", "No"],
            "Default": ["—", "—", "—", "—", "—"],
            "Description": [
                "LLM calls (Claude Haiku 4.5)",
                "Whisper STT",
                "Inworld TTS",
                "TURN server (prod WebRTC)",
                "TURN server (prod WebRTC)",
            ],
        }
    )

    st.divider()

    # ── Endpoints ─────────────────────────────────────────────────────────────────
    st.header("Endpoints")

    st.markdown("### `POST /sessions`")
    st.write("Create a session. Returns `session_id` and `stream_url`.")
    st.code(
        """\
curl -X POST http://localhost:8000/sessions \\
  -H "Content-Type: application/json" \\
  -d '{
    "speech": { "provider": "inworld", "model": "inworld-tts-1", "voice": "Craig" },
    "case": {
      "demographics": {
        "name": "John Smith",
        "date_of_birth": "1965-04-12",
        "sex": "male",
        "gender": "man",
        "background": "White, English-speaking, from rural Ohio"
      },
      "chief_concern": "chest pain",
      "free_information": ["The pain started two hours ago."],
      "locked_information": ["I have a history of hypertension."]
    },
    "time_limit": 300
  }'""",
        language="bash",
    )
    st.code(
        """\
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "stream_url": "/sessions/3fa85f64-5717-4562-b3fc-2c963f66afa6/stream",
  "expires_in": 300
}""",
        language="json",
    )

    st.markdown("### Connect Audio")
    st.write(
        "Both routes are auto-mounted by FastRTC after `POST /sessions`. "
        "Use either WebRTC or WebSocket."
    )

    st.subheader("WebRTC — `POST {stream_url}/webrtc/offer`")
    st.write("Send an SDP offer, receive an SDP answer.")
    st.code(
        """\
curl -X POST http://localhost:8000/sessions/<session_id>/stream/webrtc/offer \\
  -H "Content-Type: application/json" \\
  -d '{ "sdp": "<SDP offer string>", "type": "offer" }'

# Response — 200 OK
{ "sdp": "<SDP answer string>", "type": "answer" }""",
        language="bash",
    )

    st.subheader("WebSocket — `WS {stream_url}/websocket/offer`")
    st.write("Stream binary PCM audio frames.")
    st.code(
        """\
// JavaScript example
const ws = new WebSocket(
  "ws://localhost:8000/sessions/<session_id>/stream/websocket/offer"
);

ws.onopen = () => {
  // Send raw PCM audio frames as binary messages
  ws.send(audioChunk);
};

ws.onmessage = (event) => {
  // Receive patient audio response as binary
  playAudio(event.data);
};""",
        language="javascript",
    )

    st.markdown("### `POST /sessions/{session_id}/end`")
    st.write("End session. Returns full transcript and duration.")
    st.code(
        """\
curl -X POST http://localhost:8000/sessions/<session_id>/end""",
        language="bash",
    )
    st.code(
        """\
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "transcript": [
    { "role": "student", "content": "What brings you in today?", "timestamp": "2026-03-06T14:00:00Z" },
    { "role": "patient",  "content": "I've had chest pain since this morning.", "timestamp": "2026-03-06T14:00:04Z" }
  ],
  "duration_seconds": 142
}""",
        language="json",
    )

    st.divider()

    # ── Input Models ──────────────────────────────────────────────────────────────
    st.header("Input Models")

    st.subheader("CreateSessionRequest")
    st.code(
        """\
{
  "speech": SpeechConfig,
  "case": PatientCase,
  "time_limit": 300
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["speech", "case", "time_limit"],
            "Type": ["SpeechConfig", "PatientCase", "int"],
            "Required": ["Yes", "Yes", "No"],
            "Description": [
                "TTS provider, model, and voice",
                "Patient case definition",
                "Session time limit in seconds (default: 300)",
            ],
        }
    )

    st.subheader("SpeechConfig")
    st.code(
        """\
{
  "provider": "inworld",
  "model": "inworld-tts-1",
  "voice": "Craig"
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["provider", "model", "voice"],
            "Type": ["\"inworld\" | \"kokoro\"", "str", "str"],
            "Required": ["Yes", "Yes", "Yes"],
            "Description": [
                "TTS provider",
                "TTS model identifier",
                "Voice name (see registry below)",
            ],
        }
    )
    st.write("**Voice Registry**")
    st.table(
        {
            "Provider": ["inworld", "kokoro"],
            "Model": ["inworld-tts-1", "kokoro"],
            "Voices": [
                "Alex, Ashley, Craig, Deborah, Dennis, Edward, Mark, Olivia, Sarah, Theodore, Timothy, Wendy",
                "af_sarah, am_puck, af_bella, etc.",
            ],
        }
    )

    st.subheader("PatientCase")
    st.code(
        """\
{
  "demographics": Demographics,
  "chief_concern": "chest pain since this morning",
  "free_information": ["The pain started about two hours ago."],
  "locked_information": ["I have a history of hypertension."],
  "behavior": "Anxious but cooperative"
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["demographics", "chief_concern", "free_information", "locked_information", "behavior"],
            "Type": ["Demographics", "str", "str[]", "str[]", "str"],
            "Required": ["Yes", "Yes", "Yes", "Yes", "No"],
            "Description": [
                "Patient demographics",
                "Presenting complaint",
                "Information the patient volunteers freely",
                "Information revealed only when directly asked",
                "Personality or behavioural notes for the AI patient",
            ],
        }
    )

    st.subheader("Demographics")
    st.code(
        """\
{
  "name": "John Smith",
  "date_of_birth": "1965-04-12",
  "sex": "male",
  "gender": "man",
  "background": "White, English-speaking, from rural Ohio"
}""",
        language="json",
    )
    st.table(
        {
            "Field": ["name", "date_of_birth", "sex", "gender", "background"],
            "Type": ["str", "str", "\"male\"|\"female\"|\"intersex\"", "\"man\"|\"woman\"|\"non-binary\"", "str"],
            "Required": ["Yes", "Yes", "Yes", "Yes", "Yes"],
            "Description": [
                "Patient's full name",
                "Date of birth (YYYY-MM-DD)",
                "Biological sex",
                "Gender identity",
                "Cultural and social background",
            ],
        }
    )

    st.divider()

    # ── Output Models ─────────────────────────────────────────────────────────────
    st.header("Output Models")

    st.subheader("CreateSessionResponse")
    st.code(
        """\
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "stream_url": "/sessions/3fa85f64-5717-4562-b3fc-2c963f66afa6/stream",
  "expires_in": 300
}""",
        language="json",
    )

    st.subheader("EndSessionResponse")
    st.code(
        """\
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "transcript": [ TranscriptMessage, ... ],
  "duration_seconds": 142
}""",
        language="json",
    )

    st.subheader("TranscriptMessage")
    st.code(
        """\
{
  "role": "student" | "patient",
  "content": "What brings you in today?",
  "timestamp": "2026-03-06T14:00:05Z"
}""",
        language="json",
    )

    st.divider()

    # ── Full Example ──────────────────────────────────────────────────────────────
    st.header("Full Example")

    with st.expander("Request"):
        st.code(
            """\
# 1. Create a session
curl -X POST http://localhost:8000/sessions \\
  -H "Content-Type: application/json" \\
  -d '{
    "speech": { "provider": "inworld", "model": "inworld-tts-1", "voice": "Craig" },
    "case": {
      "demographics": {
        "name": "John Smith",
        "date_of_birth": "1965-04-12",
        "sex": "male",
        "gender": "man",
        "background": "White, English-speaking, from rural Ohio"
      },
      "chief_concern": "chest pain since this morning",
      "free_information": [
        "The pain started about two hours ago.",
        "It feels like pressure in the centre of my chest."
      ],
      "locked_information": [
        "I have a history of hypertension.",
        "My father had a heart attack at 58."
      ],
      "behavior": "Anxious but cooperative. Tends to downplay symptoms."
    },
    "time_limit": 300
  }'

# 2. End the session
curl -X POST http://localhost:8000/sessions/3fa85f64-5717-4562-b3fc-2c963f66afa6/end""",
            language="bash",
        )

    with st.expander("Response"):
        st.code(
            """\
# POST /sessions — 200 OK
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "stream_url": "/sessions/3fa85f64-5717-4562-b3fc-2c963f66afa6/stream",
  "expires_in": 300
}

# POST /sessions/{session_id}/end — 200 OK
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "transcript": [
    { "role": "student", "content": "What brings you in today?",             "timestamp": "2026-03-06T14:00:00Z" },
    { "role": "patient",  "content": "I've had chest pain since this morning.", "timestamp": "2026-03-06T14:00:04Z" },
    { "role": "student", "content": "Can you describe the pain?",            "timestamp": "2026-03-06T14:00:08Z" },
    { "role": "patient",  "content": "It feels like pressure in my chest.",   "timestamp": "2026-03-06T14:00:12Z" }
  ],
  "duration_seconds": 142
}""",
            language="json",
        )

    st.divider()
    st.button("Back", on_click=set_stage, args=["MAIN"], use_container_width=True, type="primary")



elif st.session_state['stage'] == "EVAL_DEMO":
    st.title("Evaluation Demo")
    st.write("This demo uses a sample patient, transcript, and rubric. You may try evaluating your own responses or evaluate one of the sample test cases.")
    st.write("NOTE: evaluation can take a few minutes.")
    st.divider()

    rubric_data = _load_rubric()
    transcript_data = _load_transcript()

    # 2 column full width layout
    layout1 = st.columns([1, 1])

    # User inputs
    summary = layout1[0].text_area(label = "**Summary Statement:** Provide a concise summary statement that uses semantic vocabulary to highlight the most important elements from history and exam to interpret and represent the patient’s main problem.", 
                                   height = 200)
    assessment = layout1[0].text_area(label = "**Assessment**: Provide a differential diagnosis and explain the reasoning behind each diagnosis.", 
                                      height = 200)
    plan = layout1[0].text_area(label = "**Plan**: Include a diagnostic plan that explains the rationale for your decision.", 
                                height = 200)

    # Transcript
    layout1[1].write("**Transcript**:")
    chat_container = layout1[1].container(height=400)
    for msg in transcript_data:
        with chat_container:
            with st.chat_message(msg["role"].capitalize()):
                st.markdown(msg['content'])
    # Physical Examination
    with layout1[1].expander("**Physical Examination**"):
        physical_exam_doc = Document("resources/physical_exam.docx")
        for paragraph in physical_exam_doc.paragraphs:
            st.write(paragraph.text)
    with layout1[1].expander("**ECG**"):
        st.image("resources/ecg.png")

    st.divider()
    layout2 = st.columns([1, 1, 1, 1, 1])

    # Evaluate
    user_responses = {
        "Summary Statement": summary, 
        "Assessment": assessment, 
        "Plan": plan
    }
    if layout2[0].button("Evaluate", use_container_width=True): 
        if any(v.strip() == "" for v in user_responses.values()):
            st.warning("Please fill in all response fields before evaluating.")
        else:
            rubric_obj = Rubric(**rubric_data)
            transcript_obj = [TranscriptMessage(**m) for m in transcript_data]
            with st.spinner("Running evaluation..."):
                result = asyncio.run(evaluate(rubric_obj, transcript_obj, user_responses))
            st.session_state["responses"] = user_responses
            st.session_state["evaluation"] = result
            set_stage("EVAL_RESULT")
            st.rerun()

    # Test cases
    layout21 = layout2[1].columns([1, 1])
    if layout21[0].button("TEST: BAD", use_container_width=True):
        with open("resources/test_case_bad.json", "r", encoding="utf8") as bad_json:
            bad_case = json.load(bad_json)
            test_responses = {
                "Summary Statement": bad_case['Summary Statement'],
                "Assessment": bad_case['Assessment'],
                "Plan": bad_case['Plan']
            }
        rubric_obj = Rubric(**rubric_data)
        transcript_obj = [TranscriptMessage(**m) for m in transcript_data]
        with st.spinner("Running evaluation..."):
            result = asyncio.run(evaluate(rubric_obj, transcript_obj, test_responses))
        st.session_state["responses"] = test_responses
        st.session_state["evaluation"] = result
        set_stage("EVAL_RESULT")
        st.rerun()

    if layout21[1].button("TEST: GOOD", use_container_width=True):
        with open("resources/test_case_good.json", "r", encoding="utf8") as good_json:
            good_case = json.load(good_json)
            test_responses = {
                "Summary Statement": good_case['Summary Statement'],
                "Assessment": good_case['Assessment'],
                "Plan": good_case['Plan']
            }
        rubric_obj = Rubric(**rubric_data)
        transcript_obj = [TranscriptMessage(**m) for m in transcript_data]
        with st.spinner("Running evaluation..."):
            result = asyncio.run(evaluate(rubric_obj, transcript_obj, test_responses))
        st.session_state["responses"] = test_responses
        st.session_state["evaluation"] = result
        set_stage("EVAL_RESULT")
        st.rerun()
    
    st.button("Back", on_click=set_stage, args=["MAIN"])


elif st.session_state['stage'] == "EVAL_RESULT":
    ev = st.session_state.get("evaluation")
    if ev is None:
        set_stage("MAIN")
        st.rerun()

    st.title("Evaluation Results")

    responses = st.session_state.get("responses", {})
    rubric_data = _load_rubric()
    rubric_items = {item["name"]: item for item in rubric_data["response_items"]}
    transcript_data = _load_transcript()

    tab_responses, tab_transcript = st.tabs(["Responses", "Transcript"])

    with tab_responses:
        for name, item in ev.responses.items():
            with st.container(border=True):
                st.subheader(name, divider="grey")
                col_left, col_right = st.columns([1, 1])
                with col_left:
                    st.markdown("**Your answer:**")
                    st.write(responses.get(item.response, ""))
                with col_right:
                    st.markdown("**Evaluation:**")
                    rubric_item = rubric_items.get(name, {})
                    if item.features:
                        for feat_key, feat_val in item.features.items():
                            icon = "✅" if feat_val["satisfied"] else "❌"
                            feat_desc = rubric_item.get("features", {}).get(feat_key, feat_key)
                            st.markdown(f"**{icon} {feat_key} — {feat_desc}**")
                            st.caption(feat_val["rationale"])
                    if item.scoring:
                        st.metric("Score", item.scoring["score"])
                        st.caption(item.scoring["rationale"])

    with tab_transcript:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("Transcript")
            with st.container(height=500):
                for message in transcript_data:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        with col_right:
            st.subheader("Evaluation")
            for label, item in ev.transcript.items():
                icon = "✅" if item.satisfied else "❌"
                st.markdown(f"**{icon} {label}**")
                st.caption(item.rationale)

    st.divider()
    layout1 = st.columns([1, 1, 5])
    layout1[0].button("Start Over", on_click=set_stage, args=["EVAL_DEMO"], use_container_width=True)
    layout1[1].button("Back to Main", on_click=set_stage, args=["MAIN"], use_container_width=True)
