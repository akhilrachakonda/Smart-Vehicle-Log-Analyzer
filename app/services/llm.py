import json
import logging
import os
from typing import Any, Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)


def _extract_json_block(text: str) -> str:
    """Strip markdown fences (```json ... ```) if the model includes them."""
    if "```" not in text:
        return text

    segments = text.split("```")
    if len(segments) < 2:
        return text

    candidate = segments[1]
    if candidate.lower().startswith("json"):
        candidate = candidate[4:]
    return candidate.strip()


def generate_explanation_with_llm(anomaly_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured, human-readable anomaly explanation using OpenAI ChatCompletion.

    Args:
        anomaly_context (Dict[str, Any]): Context containing sensor values, anomaly score, timestamp, and feature name.

    Returns:
        Dict[str, Any]: Dictionary with keys 'root_cause', 'severity', and 'recommended_actions'.

    Note:
        FastAPI integration: Call this function inside your anomaly detection route and merge the returned
        dictionary into the response payload for each detected anomaly.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not configured; returning fallback LLM explanation.")
        return {
            "root_cause": "LLM explanation unavailable (missing API key).",
            "severity": "MEDIUM",
            "recommended_actions": ["Configure OPENAI_API_KEY and retry LLM explanation generation."]
        }

    client = OpenAI(api_key=api_key)

    fallback_response = {
        "root_cause": "Automatic explanation unavailable. Refer to raw anomaly context.",
        "severity": "MEDIUM",
        "recommended_actions": ["Review the anomaly context and rerun explanation generation later."]
    }

    # Sample prompt sent to the model; kept explicit for observability.
    prompt = (
        "You are an automotive diagnostics assistant. Given the anomaly context, provide:\n"
        "1) A concise root cause hypothesis.\n"
        "2) Severity category as one of Low, Medium, or High.\n"
        "3) 2-3 recommended actions a vehicle technician can take next.\n\n"
        "Respond ONLY with valid JSON using keys: root_cause (string), severity (Low|Medium|High), "
        "recommended_actions (array of strings).\n\n"
        f"Anomaly context:\n{json.dumps(anomaly_context, indent=2)}"
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert vehicle diagnostics assistant. Return structured JSON only."
        },
        {"role": "user", "content": prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        cleaned = _extract_json_block(content).strip()
        parsed = json.loads(cleaned)

        root_cause = str(parsed.get("root_cause", "")).strip() or fallback_response["root_cause"]
        severity = str(parsed.get("severity", "Medium")).capitalize()
        if severity not in {"Low", "Medium", "High"}:
            severity = "Medium"
        severity = severity.upper()

        recommended_actions_raw = parsed.get("recommended_actions", [])
        if isinstance(recommended_actions_raw, str):
            recommended_actions: List[str] = [recommended_actions_raw]
        else:
            recommended_actions = [str(action).strip() for action in recommended_actions_raw if str(action).strip()]

        if not recommended_actions:
            recommended_actions = fallback_response["recommended_actions"]

        return {
            "root_cause": root_cause,
            "severity": severity,
            "recommended_actions": recommended_actions,
        }

    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s", exc)
    except Exception as exc:
        logger.exception("LLM explanation generation failed: %s", exc)

    return fallback_response
