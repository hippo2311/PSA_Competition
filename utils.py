"""
utils.py
Helpers: parsing, extraction, and schema definitions
"""
import io
import json
import re
from textwrap import dedent
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# ---------- Known schemas ----------
LOG_SCHEMAS = {
    "api_event_log": ["log_timestamp", "type", "details", "api_event_id", "container_id", "correlation_id"],
    "berth_application_log": ["log_timestamp", "type", "details", "system_vessel_name", "vessel_advice_no", 
                              "application_no", "correlation_id"],
    "edi_advice_log": ["log_timestamp", "type", "details", "corrId", "messageType", "httpStatus", "code"],
    "container_service_log": ["log_timestamp", "type", "details", "cntr_no", "correlation_id"],
    "vessel_advice_log": ["log_timestamp", "type", "details", "vesselName", "system_vessel_name", "corrId"],
    "vessel_registry_log": ["log_timestamp", "type", "details", "vessel_id", "imo_no", "old_flag", "new_flag", "user"],
}

DB_SCHEMAS = {
    "api_event": [
        "api_id", "container_id", "vessel_id", "event_type", "source_system", "http_status",
        "correlation_id", "event_ts", "payload_json", "created_at",
    ],
    "berth_application": [
        "application_id", "vessel_advice_no", "vessel_close_datetime", "deleted", 
        "berthing_status", "created_at",
    ],
    "container": [
        "container_id", "cntr_no", "iso_code", "size_type", "gross_weight_kg", "status",
        "origin_port", "tranship_port", "destination_port", "hazard_class", "vessel_id",
        "eta_ts", "etd_ts", "last_free_day", "created_at",
    ],
    "edi_message": [
        "edi_id", "container_id", "vessel_id", "message_type", "direction", "status",
        "message_ref", "sender", "receiver", "sent_at", "ack_at", "error_text",
        "raw_text", "created_at",
    ],
    "vessel": [
        "vessel_id", "imo_no", "vessel_name", "call_sign", "operator_name", "flag_state",
        "built_year", "capacity_teu", "loa_m", "beam_m", "draft_m", "last_port", 
        "next_port", "created_at",
    ],
    "vessel_advice": [
        "vessel_advice_no", "vessel_name", "system_vessel_name", "effective_start_datetime",
        "effective_end_datetime", "created_at", "system_vessel_name_active",
    ],
}

def dedent_sql(sql: str) -> str:
    """Clean up SQL formatting"""
    return dedent(sql).strip()


# ---------- Extraction helpers ----------
CNTR_RE = re.compile(r"\b([A-Z]{4}\d{7})\b")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
MSGREF_RE = re.compile(r"\b([A-Z]{3,4}-[A-Z]{3}-\d{4,})\b")


def extract_candidates(text: str) -> Dict[str, List[str]]:
    """Extract likely identifiers from arbitrary text"""
    text = (text or "").strip()
    cand = {
        "cntr_no": list({m.group(1) for m in CNTR_RE.finditer(text)}),
        "correlation_id": list({m.group(0) for m in UUID_RE.finditer(text)}),
        "message_ref": list({m.group(1) for m in MSGREF_RE.finditer(text)}),
        "vessel_name_like": [],
    }
    
    # Capture vessel patterns like "MV SILVER CURRENT"
    vessel_tokens = re.findall(r"\b(?:MV|MT|M/V|M\.?V\.?)\s+([A-Z0-9\- ]{3,})", 
                               text, flags=re.IGNORECASE)
    for vt in vessel_tokens:
        clean = re.sub(r"[^A-Z0-9 \-]", "", vt, flags=re.IGNORECASE).strip()
        if clean:
            cand["vessel_name_like"].append(clean)

    # Deduplicate
    cand = {k: sorted(set(v)) for k, v in cand.items()}
    return cand


# ---------- Google Sheets Integration ----------
TICKETS_SHEET_ID = "1Xfy-QQzHc34BEmyofKG-9uWcYuRCXqprIB7wCfujBzs"
CASE_LOGS_SHEET_ID = "1G5Z46AM6GiOFzxocrpd4JKdtfwbjFKBtoxRD8ZoqrKU"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_gspread_client():
    """Initialize Google Sheets client using Streamlit secrets"""
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], 
            scopes=SCOPES
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"❌ Failed to authorize Google Sheets: {e}")
        return None


def write_ticket_to_sheets(ticket_data: Dict) -> bool:
    """
    Write ticket data to Google Sheets (Tickets sheet)
    
    Expected columns:
    Ticket ID | Module | Mode | EDI? | TIMESTAMP | Alert / Email | Problem Statements | Solution | SOP
    """
    client = get_gspread_client()
    if not client:
        return False
    
    try:
        sheet = client.open_by_key(TICKETS_SHEET_ID).sheet1
        
        # Prepare row data (9 columns)
        row = [
            ticket_data.get("Ticket ID", ""),
            ticket_data.get("Module", ""),
            ticket_data.get("Mode", ""),
            ticket_data.get("EDI?", ""),
            ticket_data.get("TIMESTAMP", ""),
            ticket_data.get("Alert / Email", ""),
            ticket_data.get("Problem Statements", ""),
            ticket_data.get("Solution", ""),
            ticket_data.get("SOP", ""),
        ]
        
        # Append row to sheet
        sheet.append_row(row, value_input_option="USER_ENTERED")
        return True
    
    except Exception as e:
        st.error(f"❌ Failed to write to Tickets sheet: {e}")
        return False


def get_latest_ticket_row(sheet_id: str = TICKETS_SHEET_ID) -> Optional[List]:
    """Get the latest ticket row from Tickets sheet"""
    client = get_gspread_client()
    if not client:
        return None
    
    try:
        sheet = client.open_by_key(sheet_id).sheet1
        all_rows = sheet.get_all_values()
        
        if len(all_rows) <= 1:  # Only header or empty
            return None
        
        # Return last row (most recent ticket)
        return all_rows[-1]
    
    except Exception as e:
        st.error(f"❌ Failed to read from Tickets sheet: {e}")
        return None


def mark_ticket_resolved(ticket_row: Optional[List] = None) -> bool:
    """
    Mark ticket as resolved by copying to Case Logs sheet (without Ticket ID column)
    Appends to the last row.
    
    Case Logs columns (8 total):
    Module | Mode | EDI | TIMESTAMP | Alert / Email | Problem Statements | Solution | SOP
    """
    if not ticket_row:
        ticket_row = get_latest_ticket_row()
    
    client = get_gspread_client()
    if not client:
        return False
    
    try:
        spreadsheet = client.open_by_key(CASE_LOGS_SHEET_ID)
        case_logs_sheet = spreadsheet.worksheet("Cases")
        
        # Extract data WITHOUT Ticket ID (columns 2-9, skipping column 1)
        resolved_row = ticket_row[1:9]  # Module, Mode, EDI?, TIMESTAMP, Alert/Email, Problem, Solution, SOP
        
        # Append to the last row (after all existing data)
        case_logs_sheet.append_row(resolved_row, value_input_option="USER_ENTERED")
        
        return True
    
    except Exception as e:
        st.error(f"❌ Failed to mark ticket as resolved: {e}")
        return False


def get_tickets_sheet_url() -> str:
    """Return the URL to the Tickets Google Sheet"""
    return f"https://docs.google.com/spreadsheets/d/{TICKETS_SHEET_ID}/edit"


def get_case_logs_sheet_url() -> str:
    """Return the URL to the Case Logs Google Sheet"""
    return f"https://docs.google.com/spreadsheets/d/{CASE_LOGS_SHEET_ID}/edit"