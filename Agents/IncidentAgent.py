# Agents/IncidentAgent.py

import json
import re
import requests
from typing import Dict, Any, Optional, Union

# ---------- Configuration ----------
API_KEY = "89f81513c88549ad9decb34d56d88850"  # PSA subscription key (demo only)
BASE_URL = "https://psacodesprint2025.azure-api.net/gpt-5-mini/openai"
DEPLOYMENT = "gpt-5-mini"
API_VERSION = "2025-04-01-preview"
TIMEOUT = 30  # seconds

# ---------- Canonicalization moved here ----------
CANONICAL_SERVICE = {
    "container": "container_service",
    "container_service": "container_service",
    "container_service.log": "container_service",

    "vessel": "vessel_advice_service",
    "vessel_advice": "vessel_advice_service",
    "vessel_advice_service": "vessel_advice_service",
    "vessel_advice_service.log": "vessel_advice_service",

    "edi": "edi_advice_service",
    "edi_message": "edi_advice_service",
    "edi_advice": "edi_advice_service",
    "edi_advice_service": "edi_advice_service",
    "edi_advice_service.log": "edi_advice_service",

    "api_event": "api_event_service",
    "api_event_service": "api_event_service",
    "api_event_service.log": "api_event_service",

    "berth_application": "berth_application_service",
    "berth_application_service": "berth_application_service",
    "berth_application_service.log": "berth_application_service",
}

def _canonize_service_name(val: str) -> str:
    if not isinstance(val, str):
        return ""
    key = val.strip().lower()
    return CANONICAL_SERVICE.get(key, key.replace(".log", ""))

def coerce_log_files_value(parsed: dict) -> Union[str, list, dict, str]:
    """
    Prefer log_files > log_file > probable_table; map anything string-like to canonical *_service.
    Supports str / list[str] / dict.
    """
    if not isinstance(parsed, dict):
        return ""
    raw = parsed.get("log_files") or parsed.get("log_file") or parsed.get("probable_table")

    if isinstance(raw, str):
        return _canonize_service_name(raw)

    if isinstance(raw, list):
        return [_canonize_service_name(x) if isinstance(x, str) else x for x in raw]

    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            out[k] = _canonize_service_name(v) if isinstance(v, str) else v
        return out

    return None

# ---------- System prompt (tightened; fixed 'edi_adive_service' typo) ----------
SYSTEM_PROMPT = (
    "You are a PSA Ops Copilot. Extract ONLY the following keys as strict JSON:\n"
    "  incident_id (string), source_type (one of: email,sms,call,other),\n"
    "  entity_type (one of: edi,container,vessel,unknown),\n"
    "  entity_value (string),\n"
    "  log_files (prefer canonical service names WITHOUT '.log': "
    "    edi_advice_service, container_service, vessel_advice_service, "
    "    api_event_service, berth_application_service, unknown),\n"
    "  error_hint (string; short like 'Segment missing', 'No acknowledgment').\n"
    "Rules:\n"
    "- If the alert mentions any alias (e.g., 'edi', 'edi_message', 'edi_advice_service.log'), "
    "  map it to the canonical service (e.g., 'edi_advice_service').\n"
    "- Do not include explanations; return JSON ONLY.\n"
)

def _endpoint_url() -> str:
    return f"{BASE_URL}/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"

def _post_to_psa(body: dict) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    try:
        resp = requests.post(_endpoint_url(), headers=headers, json=body, timeout=TIMEOUT)
    except Exception as e:
        return {"error": f"Network error: {e}", "endpoint": _endpoint_url()}
    if resp.status_code != 200:
        preview = resp.text[:500].replace("\n", " ")
        return {"error": f"HTTP {resp.status_code}: {preview}", "endpoint": _endpoint_url()}
    try:
        return resp.json()
    except Exception as e:
        raw_preview = resp.text[:500].replace("\n", " ")
        return {"error": f"Invalid JSON response: {e}", "raw": raw_preview}

# ---------- Option A: function calling (preferred; avoids brittle parsing) ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "return_structured_alert",
            "description": "Return parsed alert as strict structured fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_id":   {"type": "string"},
                    "source_type":   {"type": "string", "enum": ["email", "sms", "call", "other"]},
                    "entity_type":   {"type": "string", "enum": ["edi", "container", "vessel", "unknown"]},
                    "entity_value":  {"type": "string"},
                    "log_files":     {
                        "type": "string",
                        "enum": [
                            "edi_advice_service",
                            "container_service",
                            "vessel_advice_service",
                            "api_event_service",
                            "berth_application_service",
                            "unknown"
                        ]
                    },
                    "error_hint":    {"type": "string"}
                },
                "required": ["incident_id","source_type","entity_type","entity_value","log_files","error_hint"],
                "additionalProperties": False,
            },
        },
    }
]

FORCED_TOOL_CHOICE = {"type": "function", "function": {"name": "return_structured_alert"}}

# ---------- Option B: response_format JSON schema (if your gateway supports it) ----------
RESPONSE_FORMAT_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "PSAAlertSchema",
        "schema": {
            "type": "object",
            "properties": {
                "incident_id":   {"type": "string"},
                "source_type":   {"type": "string", "enum": ["email", "sms", "call", "other"]},
                "entity_type":   {"type": "string", "enum": ["edi", "container", "vessel", "unknown"]},
                "entity_value":  {"type": "string"},
                "log_files":     {"type": "string", "enum": [
                    "edi_advice_service",
                    "container_service",
                    "vessel_advice_service",
                    "api_event_service",
                    "berth_application_service",
                    "unknown"
                ]},
                "error_hint":    {"type": "string"},
            },
            "required": ["incident_id","source_type","entity_type","entity_value","log_files","error_hint"],
            "additionalProperties": False,
        }
    }
}

# ---------- Fallback text cleaner (kept, but rarely needed with tools/schema) ----------
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
def _normalize_before_parse(s: str) -> str:
    s = _CODE_FENCE_RE.sub("", s).strip()
    if s and s[0] == "\ufeff":
        s = s[1:]
    return s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

def _extract_first_balanced_json(text: str) -> Dict[str, Any]:
    cleaned = _normalize_before_parse(text)
    start = cleaned.find("{")
    if start == -1:
        return {"error": "No JSON object found", "raw_content": cleaned[:500]}
    depth = 0; in_str = False; esc = False; end = None
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i; break
    if end is None:
        return {"error": "Unbalanced JSON braces", "raw_content": cleaned[start:start+500]}
    candidate = cleaned[start:end + 1].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        pointer = max(0, e.pos - 50)
        snippet = candidate[pointer:e.pos + 50]
        return {"error": f"JSON parse error: {e}", "snippet": snippet}

# ---------- Optional ADK integration ----------
_USE_ADK = True
try:
    from google.adk.agents import Agent  # type: ignore
except Exception:
    _USE_ADK = False
    Agent = None  # type: ignore

class PSAAlertAgent:
    """
    Parses one alert text into strict structured fields.
    Prefers: function-calling (tools) → JSON schema response_format → fallback text parser.
    """

    def __init__(self):
        self._agent = None
        if _USE_ADK and Agent is not None:
            try:
                # Minimal wrapper that will call our HTTP function internally if needed
                self._agent = Agent(
                    name="PSA_Alert_Parser",
                    description="Parses one PSA alert into strict JSON fields.",
                    tools=[]
                )
            except Exception:
                self._agent = None

    def _chat_tools(self, alert_text: str) -> Dict[str, Any]:
        """
        Use function calling with a forced tool choice. The model returns:
          choices[0].message.tool_calls[0].function.arguments  (JSON string)
        -> Load that JSON and return as dict. No free-text parsing.
        """
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": alert_text},
            ],
            "model": DEPLOYMENT,
            "tools": TOOLS,
            "tool_choice": FORCED_TOOL_CHOICE,
        }
        data = _post_to_psa(body)
        if "error" in data: return data

        try:
            tool_calls = data["choices"][0]["message"]["tool_calls"]
            args_json = tool_calls[0]["function"]["arguments"]
            obj = json.loads(args_json)
            return obj
        except Exception:
            # If tools not supported by gateway, fall back to response_format or text route
            return {"_tools_not_supported": True, "raw": data}

    def _chat_json_schema(self, alert_text: str) -> Dict[str, Any]:
        """
        Use response_format JSON schema so the model must return a JSON object conforming to the schema.
        Some gateways may not support this; we detect and fall back.
        """
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": alert_text},
            ],
            "model": DEPLOYMENT,
            "response_format": RESPONSE_FORMAT_JSON_SCHEMA,
        }
        data = _post_to_psa(body)
        if "error" in data: return data
        try:
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)  # should be a perfect JSON object
        except Exception:
            return {"_json_schema_not_supported": True, "raw": data}

    def _chat_text_fallback(self, alert_text: str) -> Dict[str, Any]:
        """
        Old path: ask for JSON in the prompt, then extract with balanced-brace finder.
        """
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": alert_text},
            ],
            "model": DEPLOYMENT,
        }
        data = _post_to_psa(body)
        if "error" in data: return data
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            try:
                content = data["choices"][0]["text"]
            except Exception:
                return {"error": "Unexpected response shape", "raw": data}
        return _extract_first_balanced_json(content)

    def run(self, alert_text: str) -> Dict[str, Any]:
        if not alert_text or not alert_text.strip():
            return {"error": "Empty alert text."}

        # 1) Try tools/function-calling
        obj = self._chat_tools(alert_text)
        if isinstance(obj, dict) and not obj.get("_tools_not_supported"):
            # Extra safety: canonize the log_files (just in case)
            if "log_files" in obj and isinstance(obj["log_files"], str):
                obj["log_files"] = _canonize_service_name(obj["log_files"])
            return obj

        # 2) Try response_format JSON schema
        obj = self._chat_json_schema(alert_text)
        if isinstance(obj, dict) and not obj.get("_json_schema_not_supported"):
            if "log_files" in obj and isinstance(obj["log_files"], str):
                obj["log_files"] = _canonize_service_name(obj["log_files"])
            return obj

        # 3) Fallback to text + parser
        obj = self._chat_text_fallback(alert_text)
        if isinstance(obj, dict) and "error" not in obj:
            # Canonicalize if model used aliases
            if "log_files" in obj:
                obj["log_files"] = _canonize_service_name(obj["log_files"]) if isinstance(obj["log_files"], str) else obj["log_files"]
        return obj
