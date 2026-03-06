"""
Self-contained evaluation logic for the demo API.

Ports evaluation_service.py with no database or simulation dependencies.
Config from env vars: OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

import aiohttp

from models import (
    Rubric,
    Evaluation,
    ResponseEvaluation,
    ResponseRubricItem,
    TranscriptEvaluation,
    TranscriptMessage,
    TranscriptRubricItem,
)

logger = logging.getLogger(__name__)

RESOURCES_DIR = Path(__file__).parent / "resources"

_template_cache: Dict[str, str] = {}


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return key


def _get_config() -> tuple[str, str, str]:
    api_key = _get_api_key()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")
    return api_key, base_url, model


async def _call_llm(system_prompt: str, user_message: str) -> str:
    api_key, base_url, model = _get_config()
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["message"]["content"]


async def _call_llm_with_retry(system_prompt: str, user_message: str, max_retries: int = 1) -> str:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await _call_llm(system_prompt, user_message)
        except (aiohttp.ClientError, aiohttp.ServerTimeoutError, TimeoutError) as e:
            last_error = e
            logger.warning(f"LLM call attempt {attempt + 1}/{max_retries + 1} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying LLM call (attempt {attempt + 2})...")
            else:
                raise Exception(f"LLM call failed after {max_retries + 1} attempts") from e
    raise Exception("LLM call failed") from last_error


def _load_template(filename: str) -> str:
    if filename not in _template_cache:
        path = RESOURCES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Evaluation template not found: {path}")
        _template_cache[filename] = path.read_text()
    return _template_cache[filename]


def _format_rubric_json(item: ResponseRubricItem) -> str:
    if item.type == "feature-based":
        return json.dumps({"features": item.features}, indent=2)
    elif item.type == "score-based":
        return json.dumps({"scoring": item.scoring}, indent=2)
    else:  # mixed
        return json.dumps({"features": item.features, "scoring": item.scoring}, indent=2)


def _build_response_prompt(item: ResponseRubricItem) -> str:
    template_type = item.type.replace("-", "_")
    template = _load_template(f"evaluation_response_{template_type}.txt")
    rubric_json = _format_rubric_json(item)
    prompt = template.replace("{{name}}", item.name)
    prompt = prompt.replace("{{description}}", item.description)
    prompt = prompt.replace("{{rubric}}", rubric_json)
    return prompt


def _build_response_user_message(item: ResponseRubricItem, student_responses: Dict[str, str]) -> str:
    parts = []
    if item.response in student_responses:
        parts.append(f"Student's {item.response}:\n{student_responses[item.response]}")
    for name in item.context_responses:
        if name in student_responses:
            parts.append(f"Student's {name} (for context):\n{student_responses[name]}")
    return "\n\n".join(parts)


def _build_transcript_prompt(item: TranscriptRubricItem) -> str:
    template = _load_template("evaluation_transcript.txt")
    prompt = template.replace("{{category}}", item.category)
    prompt = prompt.replace("{{label}}", item.label)
    prompt = prompt.replace("{{description}}", item.description)
    return prompt


def _build_transcript_user_message(transcript: List[TranscriptMessage]) -> str:
    if not transcript:
        return "No transcript available."
    lines = [f"{msg.role.capitalize()}: {msg.content}" for msg in transcript]
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()
    return cleaned


def _parse_response_llm_output(response_text: str, item_type: str) -> Dict:
    cleaned = _strip_fences(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}\n{response_text[:500]}")
        raise ValueError(f"Invalid JSON in LLM response: {e}") from e

    if item_type == "feature-based":
        data.pop("scoring", None)
        for name, val in data.items():
            if not isinstance(val, dict) or "rationale" not in val or "satisfied" not in val:
                raise ValueError(f"Feature '{name}' must have 'rationale' and 'satisfied' keys")
    elif item_type == "score-based":
        if "scoring" not in data:
            raise ValueError("Score-based evaluation must contain 'scoring' key")
        scoring = data["scoring"]
        if "rationale" not in scoring or "score" not in scoring:
            raise ValueError("Scoring dict must contain 'rationale' and 'score' keys")
    else:  # mixed
        if "scoring" not in data:
            raise ValueError("Mixed evaluation must contain 'scoring' key")
        scoring = data["scoring"]
        if "rationale" not in scoring or "score" not in scoring:
            raise ValueError("Scoring dict must contain 'rationale' and 'score' keys")
        for name, val in {k: v for k, v in data.items() if k != "scoring"}.items():
            if not isinstance(val, dict) or "rationale" not in val or "satisfied" not in val:
                raise ValueError(f"Feature '{name}' must have 'rationale' and 'satisfied' keys")

    return data


def _parse_transcript_llm_output(response_text: str) -> Dict:
    cleaned = _strip_fences(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse transcript LLM response as JSON: {e}\n{response_text[:500]}")
        raise ValueError(f"Invalid JSON in LLM response: {e}") from e

    if "rationale" not in data or "satisfied" not in data:
        raise ValueError("Transcript evaluation must contain 'rationale' and 'satisfied' keys")

    return data


async def _evaluate_transcript_item(
    item: TranscriptRubricItem,
    transcript: List[TranscriptMessage],
) -> TranscriptEvaluation:
    system_prompt = _build_transcript_prompt(item)
    user_message = _build_transcript_user_message(transcript)
    response_text = await _call_llm_with_retry(system_prompt, user_message)
    parsed = _parse_transcript_llm_output(response_text)
    return TranscriptEvaluation(rationale=parsed["rationale"], satisfied=parsed["satisfied"])


async def _evaluate_response_item(
    item: ResponseRubricItem,
    student_responses: Dict[str, str],
) -> ResponseEvaluation:
    if item.response not in student_responses:
        logger.warning(f"Response '{item.response}' not found in student responses")
        return ResponseEvaluation(response=item.response, features={}, scoring=None, feedback="")

    system_prompt = _build_response_prompt(item)
    user_message = _build_response_user_message(item, student_responses)
    response_text = await _call_llm_with_retry(system_prompt, user_message)
    parsed = _parse_response_llm_output(response_text, item.type)

    if item.type == "feature-based":
        return ResponseEvaluation(response=item.response, features=parsed, scoring=None, feedback="")
    elif item.type == "score-based":
        return ResponseEvaluation(response=item.response, features={}, scoring=parsed["scoring"], feedback="")
    else:  # mixed
        return ResponseEvaluation(
            response=item.response,
            features={k: v for k, v in parsed.items() if k != "scoring"},
            scoring=parsed["scoring"],
            feedback="",
        )


async def evaluate(
    rubric: Rubric,
    transcript: List[TranscriptMessage],
    responses: Dict[str, str],
) -> Evaluation:
    transcript_evaluations: Dict[str, TranscriptEvaluation] = {}
    for item in rubric.transcript_items:
        try:
            transcript_evaluations[item.label] = await _evaluate_transcript_item(item, transcript)
            logger.info(f"Evaluated transcript item: {item.label}")
        except Exception as e:
            logger.error(f"Failed to evaluate transcript item '{item.label}': {e}")
            raise Exception(f"Evaluation failed for transcript item '{item.label}': {e}") from e

    response_evaluations: Dict[str, ResponseEvaluation] = {}
    for item in rubric.response_items:
        try:
            response_evaluations[item.name] = await _evaluate_response_item(item, responses)
            logger.info(f"Evaluated response item: {item.name}")
        except Exception as e:
            logger.error(f"Failed to evaluate response item '{item.name}': {e}")
            raise Exception(f"Evaluation failed for response item '{item.name}': {e}") from e

    return Evaluation(transcript=transcript_evaluations, responses=response_evaluations)
