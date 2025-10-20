# Agents/ExtractorAgent.py
# --------------------------------------------------------------------
# Extractor AI (PSA – Problem 3, PORTNET® L2 Product Ops)
# - Inputs (max 3):
#   1) historical_data: dict
#   2) knowledge_base: dict
#   3) judge_output: str|dict|list
# - Auto-loads Product_TeamEscalation_Contacts.pdf (overrideable / or paste text)
# - Returns Markdown TEXT by default (pretty), or dict if return_dict=True
# - Generates 3 documents (PDF if ReportLab available, else .txt):
#   * Knowledge Base AI solution
#   * Historical Data AI solution
#   * Our Own AI solution
# - Resilient networking with retry/backoff and an OFFLINE FALLBACK
# - PDFs are structured into:
#     Section 1 – Problem
#     Section 2 – RCA
#     Section 3 – Solution (step-by-step)
# - Text output shows ONLY:
#     Executive Summary (bold heading), Alert message, Problem, RCA,
#     Best solution summary & high-level steps (not detailed),
#     Escalation (Name – Role – Contact), Reason to contact,
#     + Download links for the 3 generated documents.
# --------------------------------------------------------------------

from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union, Tuple

import requests
import datetime
from zoneinfo import ZoneInfo
import pandas as pd

# ---------- Optional backends ----------
try:
    import pdfplumber  # reading contacts
except ImportError:
    pdfplumber = None

try:
    # writing PDFs (fallback to .txt if missing)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    _HAS_PDF_WRITE = True
except Exception:
    _HAS_PDF_WRITE = False

# ----------------------------- Config -------------------------------
API_KEY = "89f81513c88549ad9decb34d56d88850"  # demo only
BASE_URL = "https://psacodesprint2025.azure-api.net/gpt-5-mini/openai"
DEPLOYMENT = "gpt-5-mini"
API_VERSION = "2025-04-01-preview"

TIMEOUT = 60  # seconds (connect+read)
MAX_RETRIES = 3
BACKOFF_BASE = 0.9  # seconds

ALLOW_EMPTY_CONTACTS = True
DEFAULT_CONTACTS_PATH = os.environ.get(
    "PSA_CONTACTS_PDF", "Knowledge/Product_TeamEscalation_Contacts.pdf"
)
OUTPUT_DIR = os.environ.get("PSA_OUTPUT_DIR", "Outputs")

# -------------------------- Ticket ID Generator ------------------------
def _generate_ticket_id(prefix: str = "TCK") -> str:
    """
    Generate a unique ticket ID with format: {PREFIX}-YYYYMMDD-HHMMSS-XXX
    where XXX is a random 3-digit number for uniqueness
    """
    import random
    timestamp = datetime.datetime.now(ZoneInfo("Asia/Singapore"))
    date_part = timestamp.strftime("%Y%m%d")
    time_part = timestamp.strftime("%H%M%S")
    random_part = random.randint(100, 999)
    
    return f"{prefix}-{date_part}-{time_part}-{random_part}"

# ---------------------------- Prompt -------------------------------
SYSTEM_PROMPT = (
    "You are Extractor AI for PSA Singapore (Problem 3: PORTNET® L2 Product Ops).\n"
    "You will receive up to three inputs:\n"
    "  (1) historical_data: dict describing prior incidents, preconditions, evidence, solution & SOP.\n"
    "  (2) knowledge_base: dict from Knowledge Base AI (problem tied to KB, preconditions, evidence, solution).\n"
    "  (3) judge_output: text from Judge AI (summary/rca/steps).\n"
    "You also receive Contacts text extracted from a PDF.\n\n"
    "TASKS:\n"
    "A) Produce FINAL FIELDS (strings or arrays):\n"
    "  - alert_message\n"
    "  - problem\n"
    "  - rca\n"
    "  - best_steps: 3–6 short high-level steps (not detailed)\n"
    "  - best_solution_summary: one paragraph summarising the best overall solution(s) found by the AIs\n"
    "  - knowledge_base_ai_solution (text body)\n"
    "  - historical_data_ai_solution (text body)\n"
    "  - our_own_ai_solution (text body)\n"
    "  - escalation: {name, role, email, phone, reason}\n\n"
    "FORMAT RULES:\n"
    "- Output STRICT JSON with keys exactly: "
    "  alert_message, problem, rca, best_steps, best_solution_summary, "
    "  knowledge_base_ai_solution, historical_data_ai_solution, our_own_ai_solution, escalation.\n"
    "- Keep content concise, auditable, operationally safe.\n"
    "- Do NOT reveal chain-of-thought.\n"
)

# -------------------------- HTTP helper -----------------------------
def _endpoint_url() -> str:
    return f"{BASE_URL}/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"

def _post_to_psa(messages: list) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    body = {"messages": messages, "model": DEPLOYMENT}
    url = _endpoint_url()
    last_err: Dict[str, Any] = {}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    raw_preview = resp.text[:800].replace("\n", " ")
                    return {"error": "Invalid JSON response", "raw": raw_preview, "endpoint": url}
            # retry on transient codes
            if resp.status_code in (429, 500, 502, 503, 504):
                preview = resp.text[:800].replace("\n", " ")
                last_err = {"error": f"HTTP {resp.status_code}: {preview}", "endpoint": url}
            else:
                preview = resp.text[:800].replace("\n", " ")
                return {"error": f"HTTP {resp.status_code}: {preview}", "endpoint": url}
        except Exception as e:
            last_err = {"error": f"Network error: {e}", "endpoint": url}
        if attempt < MAX_RETRIES:
            import random
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.25))
    return last_err or {"error": "Unknown error", "endpoint": url}

# --------------------------- JSON utils -----------------------------
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)

def _extract_first_json(text: str) -> Dict[str, Any]:
    text = _CODE_FENCE_RE.sub("", (text or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"error": "No JSON found", "raw": text[:400]}
    try:
        return json.loads(text[start:end+1])
    except Exception as e:
        return {"error": f"Parse error: {e}", "raw": text[start:end+1][:400]}

# --------------------------- Contacts -------------------------------
_contacts_cache: Dict[str, str] = {}

def _resolve_pdf_path(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_CONTACTS_PATH
    return name if os.path.dirname(name) else os.path.join("Knowledge", name)

def _read_pdf_text(path: str) -> Tuple[str, bool]:
    if not pdfplumber or not os.path.exists(path):
        return ("", False)
    try:
        if path in _contacts_cache:
            return (_contacts_cache[path], True)
        with pdfplumber.open(path) as pdf:
            txt = "\n".join([p.extract_text() or "" for p in pdf.pages])
            _contacts_cache[path] = txt
            return (txt, True)
    except Exception:
        return ("", False)

def _get_contacts_text(contacts_pdf: Optional[str], contacts_text: Optional[str]) -> Tuple[str, str, bool]:
    if contacts_text and contacts_text.strip():
        return (contacts_text, "[inline]", True)
    path = _resolve_pdf_path(contacts_pdf)
    text, ok = _read_pdf_text(path)
    return (text, path, ok)

# --------------------------- File + PDF helpers ---------------------------
def _ensure_outdir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def _slug(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", (s or "").strip())[:60]
    return s or "doc"

def _wrap_line(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    line: List[str] = []
    count = 0
    for tok in text.split(" "):
        need = len(tok) + (1 if line else 0)
        if count + need > max_chars:
            parts.append(" ".join(line))
            line, count = [tok], len(tok)
        else:
            if line:
                line.append(tok); count += need
            else:
                line, count = [tok], len(tok)
    if line:
        parts.append(" ".join(line))
    return parts

def _draw_separator(c: "canvas.Canvas", x: float, y: float, width: float, color="#DDDDDD"):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(0.6)
    c.line(x, y, x + width, y)

def _draw_section_title(c: "canvas.Canvas", text: str, left: float, y: float, page_h: float, font="Helvetica-Bold", size=12) -> float:
    if y < 2*cm:
        c.showPage(); c.setFont(font, size); y = page_h - 2*cm
    c.setFont(font, size)
    c.drawString(left, y, text)
    return y - 16

def _draw_kv(c: "canvas.Canvas", key: str, val: str, left: float, y: float, page_w: float, page_h: float,
             key_font="Helvetica-Bold", key_size=11, val_font="Helvetica", val_size=11, leading=15, max_chars=95) -> float:
    if y < 2*cm:
        c.showPage(); y = page_h - 2*cm
    c.setFont(key_font, key_size)
    c.drawString(left, y, f"{key}:")
    y -= leading
    c.setFont(val_font, val_size)
    for line in (val or "").splitlines():
        if not line.strip():
            y -= leading
            if y < 2*cm:
                c.showPage(); y = page_h - 2*cm; c.setFont(val_font, val_size)
            continue
        for chunk in _wrap_line(line.strip(), max_chars=max_chars):
            if y < 2*cm:
                c.showPage(); y = page_h - 2*cm; c.setFont(val_font, val_size)
            c.drawString(left + 0.4*cm, y, chunk)
            y -= leading
    return y

def _maybe_parse_json_block(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    if s.lstrip().startswith("{") and s.rstrip().endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

# ---- NEW: normalize solution steps from any text format ----
def _normalize_steps(raw: str) -> List[str]:
    """
    Turn any 'solution' value into step strings.
    Accepts JSON-array-like, markdown bullets, or plain multiline text.
    """
    if not raw:
        return []
    # split by lines, strip bullets/numbers
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^(\d+[\.)]\s*|-|\*|\u2022)\s*", "", ln)
        lines.append(ln)
    return lines

# ---- REWORKED: render the 3-section PDF ----
def _render_solution_body_to_pdf(c: "canvas.Canvas", title: str, raw: str, left: float, top: float, page_w: float, page_h: float) -> None:
    """
    Render a clean report page:
      Section 1: Problem
      Section 2: RCA
      Section 3: Solution (step-by-step)
    Falls back to generic KV blocks when structured fields aren't available.
    """
    y = top
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, title)
    y -= 10
    _draw_separator(c, left, y, page_w - 4*cm)
    y -= 14

    # Try to parse JSON if provided
    parsed = _maybe_parse_json_block(raw)

    # Extract canonical fields (problem, rca, solution) from JSON or heuristics
    problem_txt, rca_txt, solution_raw = "", "", ""

    if parsed is not None:
        problem_txt  = str(parsed.get("problem") or parsed.get("problem_identified") or "").strip()
        rca_txt      = str(parsed.get("rca") or "").strip()
        sol_val      = parsed.get("solution")
        if isinstance(sol_val, list):
            solution_raw = "\n".join(str(x) for x in sol_val)
        else:
            solution_raw = str(sol_val or "").strip()
    else:
        # Heuristic pull from free text "Problem:", "RCA:", "Solution:" blocks
        rx = lambda key: re.search(rf"(?mi)^\s*{key}\s*:\s*(.*?)(?=^\s*[A-Z][A-Za-z ]{{2,}}:\s*|$\Z)", raw, re.S)
        m_p = rx("Problem"); m_r = rx("RCA"); m_s = rx("Solution")
        problem_txt  = (m_p.group(1).strip() if m_p else "")
        rca_txt      = (m_r.group(1).strip() if m_r else "")
        solution_raw = (m_s.group(1).strip() if m_s else "")

    # If nothing canonical, fall back to generic KV render of what we can find
    if not (problem_txt or rca_txt or solution_raw):
        # generic block rendering
        y = _draw_kv(c, "Details", raw or "(empty)", left, y, page_w, page_h, leading=15, max_chars=95)
        c.showPage()
        return

    # Section 1: Problem
    y = _draw_kv(c, "Section 1 – Problem", problem_txt or "(not specified)", left, y, page_w, page_h, leading=15, max_chars=95)

    # Section 2: RCA
    y = _draw_kv(c, "Section 2 – RCA", rca_txt or "(not specified)", left, y, page_w, page_h, leading=15, max_chars=95)

    # Section 3: Solution (step-by-step)
    steps = _normalize_steps(solution_raw)
    if not steps and solution_raw:
        # single paragraph
        y = _draw_kv(c, "Section 3 – Solution (step-by-step)", solution_raw, left, y, page_w, page_h, leading=15, max_chars=95)
    else:
        if y < 2*cm:
            c.showPage(); y = page_h - 2*cm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Section 3 – Solution (step-by-step):")
        y -= 15
        c.setFont("Helvetica", 11)
        for i, st in enumerate(steps or ["(no steps provided)"], 1):
            for chunk in _wrap_line(f"{i}. {st}", max_chars=95):
                if y < 2*cm:
                    c.showPage(); y = page_h - 2*cm; c.setFont("Helvetica", 11)
                c.drawString(left + 0.4*cm, y, chunk)
                y -= 15

    c.showPage()

def _write_pdf_or_txt_pretty(kind_title: str, body: str, basename: str) -> str:
    outdir = _ensure_outdir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    if _HAS_PDF_WRITE:
        path = os.path.join(outdir, f"{_slug(basename)}_{ts}.pdf")
        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4
        left, top = 2*cm, h - 2*cm
        _render_solution_body_to_pdf(c, kind_title, body or "", left, top, w, h)
        c.save()
        return path
    else:
        path = os.path.join(outdir, f"{_slug(basename)}_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(kind_title + "\n" + "-" * len(kind_title) + "\n\n" + (body or ""))
        return path

# --------------------- OFFLINE FALLBACK HELPERS ----------------------
def _mk_exec_summary(hist_text: str, kb_text: str, judge_text: str) -> Dict[str, Any]:
    def find_alert(s: str) -> Optional[str]:
        m = re.search(r"(ALR-\d{4,}[^\\n]*)", s or "", re.I)
        return m.group(1).strip() if m else None
    alert = find_alert(hist_text) or find_alert(kb_text) or find_alert(judge_text) or "Alert received"

    def find_problem(s: str) -> Optional[str]:
        s = s or ""
        m = re.search(r"problem_identified\"?\s*:\s*\"([^\"]+)", s, re.I)
        if m: return m.group(1)
        m = re.search(r"^Problem\s*:\s*(.+)$", s, re.I | re.M)
        if m: return m.group(1).strip()
        m = re.search(r"(duplicate|conflict|timeout|failed|error|missing|constraint)", s, re.I)
        return m.group(1) if m else None
    problem = find_problem(hist_text) or find_problem(kb_text) or find_problem(judge_text) or "Issue identified"

    def find_rca(s: str) -> Optional[str]:
        m = re.search(r"^RCA\s*:\s*(.+)$", s or "", re.I | re.M)
        return m.group(1).strip() if m else None
    rca = find_rca(judge_text) or "Pending verification"

    steps = []
    for ln in (judge_text or "").splitlines():
        if ln.strip().startswith(("- ", "* ")) or re.match(r"^\d+\.\s", ln.strip()):
            steps.append(re.sub(r"^\d+\.\s*", "", ln.strip().lstrip("-* ").strip()))
    if not steps:
        steps = ["Validate current state", "Check relevant logs", "Apply safe corrective action", "Verify outcome"]

    best_soln_summary = []
    if kb_text.strip(): best_soln_summary.append("KB solution appears applicable.")
    if hist_text.strip(): best_soln_summary.append("Historical solution with SOP found.")
    if judge_text.strip(): best_soln_summary.append("Judge AI provides high-level steps.")
    if not best_soln_summary:
        best_soln_summary.append("No AI sources available; propose safe diagnostics and remediation flow.")
    best_summary = " ".join(best_soln_summary)

    return {
        "alert_message": alert,
        "problem": problem,
        "rca": rca,
        "best_steps": steps[:6],
        "best_solution_summary": best_summary
    }

def _mk_kb_section_text(kb: Union[str, Dict[str, Any]]) -> str:
    if isinstance(kb, dict):
        data = kb
    else:
        parsed = _maybe_parse_json_block(kb)
        data = parsed if parsed is not None else {}
    if data:
        parts = []
        if "alert_message" in data: parts.append(f"Alert message: {data['alert_message']}")
        if "kb_reference" in data: parts.append(f"KB reference: {data['kb_reference']}")
        if "preconditions" in data: parts.append(f"Preconditions / Evidence: {data['preconditions']}")
        if "solution" in data:
            if isinstance(data["solution"], list):
                sol = "\n".join(f"- {s}" for s in data["solution"])
            else:
                sol = str(data["solution"])
            parts.append(f"Solution:\n{sol}")
        return "\n".join(parts)
    return (kb or "").strip()

def _mk_hist_section_text(hist: Union[str, Dict[str, Any]]) -> str:
    if isinstance(hist, dict):
        data = hist
    else:
        parsed = _maybe_parse_json_block(hist)
        data = parsed if parsed is not None else {}
    if data:
        parts = []
        if "alert_message" in data: parts.append(f"Alert message: {data['alert_message']}")
        if "problem_identified" in data: parts.append(f"Problem identified: {data['problem_identified']}")
        if "preconditions" in data: parts.append(f"Preconditions / Evidence: {data['preconditions']}")
        if "solution" in data:
            if isinstance(data["solution"], list):
                sol = "\n".join(f"- {s}" for s in data["solution"])
            else:
                sol = str(data["solution"])
            parts.append(f"Solution:\n{sol}")
        if "SOP" in data: parts.append(f"SOP: {data['SOP']}")
        return "\n".join(parts)
    return (hist or "").strip()

def _mk_own_section_text(judge: str) -> str:
    return (judge or "").strip() or "Alert: (unknown)\nProblem: (unknown)\nLogs/Data: (n/a)\nRCA: (n/a)\nSolution: (n/a)"

def _pick_contact_from_contacts(contacts_text: str) -> Dict[str, str]:
    name_role = re.search(r"^([A-Z][a-zA-Z .'-]{2,})\s*[-–]\s*([A-Za-z0-9&/() ,.-]{3,})$", contacts_text or "", re.M)
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", contacts_text or "")
    phone = re.search(r"(?:\+?\d[\d\s\-]{6,}\d)", contacts_text or "")
    return {
        "name": (name_role.group(1).strip() if name_role else ""),
        "role": (name_role.group(2).strip() if name_role else ""),
        "email": (email.group(0).strip() if email else ""),
        "phone": (phone.group(0).strip() if phone else ""),
        "reason": "Selected via offline fallback due to LLM unavailability."
    }

# ----------------------------- Agent --------------------------------
class ExtractorAgent:
    """
    Usage:
      agent = ExtractorAgent()
      text_or_json = agent.run(
        historical_data={<dict>|None},
        knowledge_base={<dict>|None},
        judge_output=<str|dict|list|None>,
        contacts_pdf="Product_TeamEscalation_Contacts.pdf",  # optional
        contacts_text=None,                                   # optional (raw text bypass)
        return_dict=False                                     # default: Markdown TEXT
      )
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _to_str(x: Union[str, Dict[str, Any], List[Any], None]) -> str:
        if x is None:
            return ""
        if isinstance(x, (dict, list)):
            try:
                return json.dumps(x, ensure_ascii=False, indent=2)
            except Exception:
                return str(x)
        return str(x).strip()

    # ---------- REWORKED: Markdown with bullets + download links ----------
    def _fmt_text_output_markdown(self, final: Dict[str, Any]) -> str:
        """
        Build Markdown with bullet points, clear line breaks, and file download links.
        """
        esc = final.get("escalation", {}) or {}
        best_steps = final.get("_best_steps_display") or []
        if not isinstance(best_steps, list):
            best_steps = []

        # File paths (show as links in Markdown UIs)
        kb_path   = final.get("knowledge_base_ai_solution_pdf", "")
        hist_path = final.get("historical_data_ai_solution_pdf", "")
        own_path  = final.get("our_own_ai_solution_pdf", "")

        def _link(label: str, path: str) -> str:
            return f"[{label}]({path})" if path else label

        md = []
        md.append("**Executive Summary**\n")
        md.append(f"- **Alert message:** {final.get('_alert_message','')}")
        md.append(f"- **Problem:** {final.get('_problem','')}")
        md.append(f"- **RCA:** {final.get('_rca','')}\n")

        md.append("**Best solution summary & high-level steps:**")
        if final.get('_best_solution_summary'):
            md.append(f"- {final['_best_solution_summary']}")
        for i, step in enumerate(best_steps, 1):
            md.append(f"  - Step {i}: {step}")
        md.append("")  # line break

        contact_str = esc.get('email') or esc.get('phone') or ""
        md.append("**Escalation**")
        md.append(f"- **Who:** {esc.get('name','')} – {esc.get('role','')}")
        if contact_str:
            md.append(f"- **Contact:** {contact_str}")
        if esc.get("reason"):
            md.append(f"- **Reason:** {esc['reason']}\n")

        md.append("**Downloads**")
        md.append(f"- {_link('Knowledge Base AI Solution', kb_path)}")
        md.append(f"- {_link('Historical Data AI Solution', hist_path)}")
        md.append(f"- {_link('Our Own AI Solution', own_path)}")

        # Optional: reveal contacts source & backend used
        if "_contacts_pdf_path" in final:
            src = final.get("_contacts_pdf_path","")
            loaded = "yes" if final.get("_contacts_loaded") else "no"
            md.append(f"\n_Contacts source: {src} (loaded: {loaded}; writer: {final.get('_pdf_backend','txt')})_")

        # Optional note if offline fallback happened
        if "_llm_error" in final and final["_llm_error"]:
            md.append(f"\n> Note: LLM unavailable, used offline fallback. ({final['_llm_error']})")

        return "\n".join(md)

    def run(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        knowledge_base: Optional[Dict[str, Any]] = None,
        judge_output: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        contacts_pdf: Optional[str] = None,
        *,
        contacts_text: Optional[str] = None,
        return_dict: bool = False,
        alert_start_ts: Optional[str] = None,
        ticket_prefix: str = "TCK",
    ) -> Union[str, Dict[str, Any]]:
        # Normalize inputs
        hist_str = self._to_str(historical_data)
        kb_str   = self._to_str(knowledge_base)
        judge_str= self._to_str(judge_output)

        # Contacts
        contacts_text_final, pdf_path, loaded = _get_contacts_text(contacts_pdf, contacts_text)
        if not contacts_text_final and not ALLOW_EMPTY_CONTACTS:
            raise RuntimeError("Contacts PDF could not be loaded and empty contacts are not allowed.")
        contacts_block = f"## CONTACTS START\n{contacts_text_final or '[EMPTY]'}\n## CONTACTS END"

        # Build user payload to LLM
        user_payload = {
            "historical_data": hist_str,
            "knowledge_base": kb_str,
            "judge_output": judge_str,
            "contacts_text": contacts_block,
            "instruction": (
                "Return STRICT JSON with keys: alert_message, problem, rca, best_steps, best_solution_summary, "
                "knowledge_base_ai_solution, historical_data_ai_solution, our_own_ai_solution, escalation."
            ),
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        # Call LLM (resilient)
        res = _post_to_psa(messages)
        
        # Generate ticket ID early (for both offline and online paths)
        ticket_id = _generate_ticket_id(ticket_prefix)
        
        if "error" in res:
            # -------- OFFLINE FALLBACK (no LLM) --------
            summary_bits = _mk_exec_summary(hist_str, kb_str, judge_str)
            alert_message = summary_bits["alert_message"]
            problem       = summary_bits["problem"]
            rca           = summary_bits["rca"]
            best_steps    = summary_bits["best_steps"]
            best_summary  = summary_bits["best_solution_summary"]

            kb_text_clean   = _mk_kb_section_text(knowledge_base if isinstance(knowledge_base, dict) else kb_str)
            hist_text_clean = _mk_hist_section_text(historical_data if isinstance(historical_data, dict) else hist_str)
            own_text_clean  = _mk_own_section_text(judge_str)

            kb_doc_path   = _write_pdf_or_txt_pretty("Knowledge Base AI Solution", kb_text_clean, "KB_AI_Solution")
            hist_doc_path = _write_pdf_or_txt_pretty("Historical Data AI Solution", hist_text_clean, "Historical_AI_Solution")
            own_doc_path  = _write_pdf_or_txt_pretty("Our Own AI Solution", own_text_clean, "Our_Own_AI_Solution")

            esc = _pick_contact_from_contacts(contacts_text_final or "")
            
            # ========== Write Ticket (Offline Path) ==========
            ticket_error = self._write_ticket(
                ticket_id=ticket_id,
                alert_message=alert_message,
                problem=problem,
                rca=rca + " (offline fallback)",
                best_summary=best_summary,
                alert_start_ts=alert_start_ts
            )

            final = {
                "_ticket_id": ticket_id,
                "_alert_message": alert_message,
                "_problem": problem,
                "_rca": rca,
                "_best_steps_display": best_steps,
                "_best_solution_summary": best_summary + " (offline fallback)",
                "knowledge_base_ai_solution_pdf": kb_doc_path,
                "historical_data_ai_solution_pdf": hist_doc_path,
                "our_own_ai_solution_pdf": own_doc_path,
                "escalation": esc,
                "_contacts_pdf_path": pdf_path,
                "_contacts_loaded": loaded,
                "_pdf_backend": "pdf" if _HAS_PDF_WRITE else "txt",
                "_llm_error": res.get("error"),
                "_ticket_error": ticket_error,
                "_ticket_csv": os.path.join(OUTPUT_DIR, "tickets_log.csv"),
            }
            if return_dict:
                return final
            return self._fmt_text_output_markdown(final)

        # Parse LLM response
        content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _extract_first_json(content)
        if "error" in parsed:
            parsed["_contacts_pdf_path"], parsed["_contacts_loaded"] = pdf_path, loaded
            parsed["_ticket_id"] = ticket_id
            return parsed if return_dict else f"[ExtractorAI] ERROR: {parsed}"

        # Pull structured fields (with defensive defaults)
        alert_message = (parsed.get("alert_message") or "").strip()
        problem       = (parsed.get("problem") or "").strip()
        rca           = (parsed.get("rca") or "").strip()
        best_steps    = parsed.get("best_steps") or []
        if not isinstance(best_steps, list): best_steps = []
        best_summary  = (parsed.get("best_solution_summary") or "").strip()

        kb_body       = (parsed.get("knowledge_base_ai_solution") or "").strip()
        hist_body     = (parsed.get("historical_data_ai_solution") or "").strip()
        own_body      = (parsed.get("our_own_ai_solution") or "").strip()

        esc           = parsed.get("escalation") or {}
        for k in ["name","role","email","phone","reason"]:
            esc.setdefault(k, "")

        # Write 3 documents (PDF/.txt) with pretty sections
        kb_doc_path   = _write_pdf_or_txt_pretty("Knowledge Base AI Solution", kb_body, "KB_AI_Solution")
        hist_doc_path = _write_pdf_or_txt_pretty("Historical Data AI Solution", hist_body, "Historical_AI_Solution")
        own_doc_path  = _write_pdf_or_txt_pretty("Our Own AI Solution", own_body, "Our_Own_AI_Solution")

        # ========== Write Ticket (Online Path) ==========
        ticket_error = self._write_ticket(
            ticket_id=ticket_id,
            alert_message=alert_message,
            problem=problem,
            rca=rca,
            best_summary=best_summary,
            alert_start_ts=alert_start_ts
        )

        final: Dict[str, Any] = {
            "_ticket_id": ticket_id,
            "_alert_message": alert_message,
            "_problem": problem,
            "_rca": rca,
            "_best_steps_display": best_steps,
            "_best_solution_summary": best_summary,
            "knowledge_base_ai_solution_pdf": kb_doc_path,
            "historical_data_ai_solution_pdf": hist_doc_path,
            "our_own_ai_solution_pdf": own_doc_path,
            "escalation": {
                "name": esc.get("name",""),
                "role": esc.get("role",""),
                "email": esc.get("email",""),
                "phone": esc.get("phone",""),
                "reason": esc.get("reason",""),
            },
            "_contacts_pdf_path": pdf_path,
            "_contacts_loaded": loaded,
            "_pdf_backend": "pdf" if _HAS_PDF_WRITE else "txt",
            "_ticket_error": ticket_error,
            "_ticket_csv": os.path.join(OUTPUT_DIR, "tickets_log.csv"),
        }

        if return_dict:
            return final
        return self._fmt_text_output_markdown(final)

    def _write_ticket(
        self,
        ticket_id: str,
        alert_message: str,
        problem: str,
        rca: str,
        best_summary: str,
        alert_start_ts: Optional[str] = None
    ) -> Optional[str]:
        """
        Write ticket to both CSV and Google Sheets.
        Returns error message if failed, None if success.
        """
        ticket_csv_path = os.path.join(OUTPUT_DIR, "tickets_log.csv")
        
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Determine module from alert message
            module = "Unknown"
            if re.search(r"EDI|COPARN|IFCSUM", alert_message or "", re.I):
                module = "EDI/API"
            elif re.search(r"vessel|berth|advice", alert_message or "", re.I):
                module = "EDI/API"
            elif re.search(r"container|CNTR", alert_message or "", re.I):
                module = "Container"
            
            # Determine mode
            mode = "Email"
            if re.search(r"SMS|INC-\d+", alert_message or "", re.I):
                mode = "SMS"
            elif re.search(r"call|phone", alert_message or "", re.I):
                mode = "Call"
            
            # Check if EDI-related
            is_edi = "Yes" if re.search(r"EDI|COPARN|IFCSUM|REF-", alert_message or "", re.I) else "No"
            
            # Timestamp - Format: M/D/YYYY H:MM:SS
            if not alert_start_ts:
                alert_start_ts = datetime.datetime.now(ZoneInfo("Asia/Singapore")).strftime("%-m/%-d/%Y %-H:%M:%S")
            else:
                # Convert ISO to display format
                try:
                    dt = datetime.datetime.fromisoformat(alert_start_ts.replace('Z', '+00:00'))
                    alert_start_ts = dt.astimezone(ZoneInfo("Asia/Singapore")).strftime("%-m/%-d/%Y %-H:%M:%S")
                except Exception:
                    pass
            
            ticket_row = {
                "Ticket ID": ticket_id,
                "Module": module,
                "Mode": mode,
                "EDI?": is_edi,
                "TIMESTAMP": alert_start_ts,
                "Alert / Email": (alert_message or "")[:500],  # Truncate for display
                "Problem Statements": (problem or "")[:500],
                "Solution": (best_summary or "")[:500],
                "SOP": rca[:200] if rca else "",  # Use RCA as SOP reference
            }
            
            # Write to CSV
            df_ticket = pd.DataFrame([ticket_row])
            if os.path.exists(ticket_csv_path):
                df_existing = pd.read_csv(ticket_csv_path)
                df_ticket = pd.concat([df_existing, df_ticket], ignore_index=True)
            df_ticket.to_csv(ticket_csv_path, index=False)
            
            try:
                from utils import write_ticket_to_sheets
                sheets_success = write_ticket_to_sheets(ticket_row)
                if not sheets_success:
                    return "Google Sheets write failed (check logs)"
            except Exception as e_sheets:
                return f"Google Sheets write failed: {e_sheets}"
            
            return None  # Success
        
        except Exception as e:
            return f"Ticket write failed: {e}"
