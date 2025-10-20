# Agents/ProposalAgent.py
# --------------------------------------------------------------------
# Judge AI:
# - Inputs: proposal_1 (str|dict|list), proposal_2 (str|dict|list)
# - Compares both Proposal AI plans, self-critiques, and picks best
#   (or merges) using internal reasoning (not revealed).
# - Returns a concise outcome with: Summary, RCA, Steps.
# --------------------------------------------------------------------

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Union

import requests

# ----------------------------- Config --------------------------------
API_KEY = "89f81513c88549ad9decb34d56d88850"  # demo-only
BASE_URL = "https://psacodesprint2025.azure-api.net/gpt-5-mini/openai"
DEPLOYMENT = "gpt-5-mini"
API_VERSION = "2025-04-01-preview"
TIMEOUT = 30  # seconds

SYSTEM_PROMPT = (
    "You are a Proposal AI working for PSA Singapore on Problem 3: "
    "AI-driven Level 2 Product Operations (PORTNET®). Your job is to produce the "
    "best actionable plan for rapid, reliable incident resolution in a high-stakes, "
    "always-on B2B port community system.\n\n"

    "Context:\n"
    "- PORTNET® connects shipping lines, hauliers, freight forwarders, and government agencies across Singapore’s maritime ecosystem.\n"
    "- Level 2 duty officers must triage multi-source incident signals (emails, forms, API alerts), correlate with system logs and historical cases, "
    "identify root cause, recommend precise remediation, and escalate to the right stakeholders fast.\n"
    "- Success is measured by business continuity and reliability outcomes: MTTR↓, false escalation↓, accurate RCA, clean handover, and resilient follow-ups.\n\n"

    "Input You Will Receive:\n"
    "You will be given outputs from two Judge AIs:\n"
    " - proposal_1\n"
    " - proposal_2\n"
    "Each proposal uses this schema (keys may vary but keep the spirit):\n"
    "{\n"
    "  \"problem_1\": {\"logs\": [...], \"solution\": [...], \"verification\": [...]},\n"
    "  \"problem_2\": {\"logs\": [...], \"solution\": [...], \"verification\": [...]}\n"
    "}\n\n"

    "Your Job:\n"
    "1) Analyze both proposals for correctness, safety, completeness, and operational impact.\n"
    "2) Use disciplined self-critique to select the better plan or synthesize a superior hybrid that is executable in PSA’s environment.\n"
    "3) Output a concise, business-readable decision with:\n"
    "   - Summary (1–2 sentences)\n"
    "   - RCA (one-line root cause)\n"
    "   - Steps (3–8 ordered, crisp, executable actions)\n\n"

    "Guidelines (PSA-L2 specific):\n"
    "- Scope & Priorities: Optimize for MTTR reduction, correctness of RCA, safety/PII handling, and clean escalation. Prefer deterministic, auditable actions.\n"
    "- Evidence-driven: Recommendations must be supported by the provided logs, prior case patterns, and KB references in the proposals’ content.\n"
    "- Safety & Compliance: Redact PII in summaries; avoid destructive actions without pre-checks/rollbacks; ensure read-only diagnostics precede writes.\n"
    "- Escalation Quality: Include who/when/how (teams/roles, not names) with readiness checks (on-call/coverage), and attach minimal diagnostic bundle.\n"
    "- Verification: Every fix must have post-change checks (health probes/log signatures/latency & error-rate deltas) and a clear rollback trigger.\n"
    "- Robustness: Call out gaps (missing logs, conflicting signals). If data is insufficient, recommend the smallest next diagnostic that unblocks a decision.\n"
    "- Reusability: Prefer actions that can be templatized (runbooks, queries, health checks) and slotted into ticketing pipelines (e.g., Jira/ServiceNow) later.\n"
    "- Clarity: Use short, imperative steps. Avoid model internals; focus on the operational plan.\n\n"

    "Non-Goals:\n"
    "- Do not expose internal chain-of-thought or model weights.\n"
    "- Do not speculate fixes that require unavailable access or tools without stating the dependency.\n\n"

    "Output Format (STRICT JSON with exactly these keys):\n"
    "{\n"
    "  \"summary\": string,\n"
    "  \"rca\": string,\n"
    "  \"steps\": [string, ...]\n"
    "}\n\n"

    "Style:\n"
    "- Do NOT mention which proposal you picked.\n"
    "- Keep it factual, concise, and action-oriented—like a PSA Product Ops post-incident summary.\n"
    "- Prefer language that an L2 duty officer can execute without further clarification.\n"
)


# ------------------------- HTTP utilities -----------------------------
def _endpoint_url() -> str:
    return f"{BASE_URL}/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"

def _post_to_psa(messages: list) -> Dict[str, Any]:
    """Send messages to the PSA gateway; return parsed JSON or a compact error dict."""
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    body = {"messages": messages, "model": DEPLOYMENT}
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
        return {"error": f"Invalid JSON response: {e}", "raw": resp.text[:500].replace('\n', ' ')}

# ----------------------- Model JSON extraction ------------------------
def _extract_first_json(text: str) -> Dict[str, Any]:
    """Strip code-fences and extract the first top-level JSON object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"error": "Could not parse JSON from model output", "raw_content": text[:500]}
    try:
        return json.loads(text[start:end+1])
    except Exception as e:
        return {"error": f"JSON parse error: {e}", "raw_content": text[start:end+1][:500]}

# ---------------------------- Public API ------------------------------
class ProposalAgent:
    """
    .run(proposal_1, proposal_2) -> dict
      - proposal_1 / proposal_2: str | dict | list (JSON-serialized for the model)
      - Returns dict with keys: summary, rca, steps
    """

    @staticmethod
    def _to_text(x: Union[str, Dict[str, Any], List[Any]]) -> str:
        if isinstance(x, str):
            return x.strip()
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)
        except Exception:
            return str(x)

    def run(
        self,
        proposal_1: Union[str, Dict[str, Any], List[Any]],
        proposal_2: Union[str, Dict[str, Any], List[Any]],
    ) -> Dict[str, Any]:
        p1, p2 = self._to_text(proposal_1), self._to_text(proposal_2)

        user_payload = {"proposal_1": p1, "proposal_2": p2}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        data = _post_to_psa(messages)
        if "error" in data:
            return data

        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
            or data.get("choices", [{}])[0].get("text", "")
        )
        return _extract_first_json(content)
