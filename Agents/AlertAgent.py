"""
Agents/AlertAgent.py
Core agent using LLM for SQL generation and analysis decisions
"""
from __future__ import annotations
import json
import requests
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

import pandas as pd

from utils import (
    DB_SCHEMAS,
    LOG_SCHEMAS,
    dedent_sql,
    extract_candidates,
)

# ---------- LLM Configuration ----------
API_KEY = "89f81513c88549ad9decb34d56d88850"
BASE_URL = "https://psacodesprint2025.azure-api.net/gpt-5-mini/openai"
DEPLOYMENT = "gpt-5-mini"
API_VERSION = "2025-04-01-preview"
TIMEOUT = 120

def _endpoint_url() -> str:
    return f"{BASE_URL}/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"

def _post_to_llm(messages: List[Dict], response_format: Optional[Dict] = None) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    body = {
        "messages": messages,
        "model": DEPLOYMENT,
    }
    if response_format:
        body["response_format"] = response_format
    
    try:
        resp = requests.post(_endpoint_url(), headers=headers, json=body, timeout=TIMEOUT)
    except Exception as e:
        return {"error": f"Network error: {e}"}
    
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
    
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {"content": content}
    except Exception as e:
        return {"error": f"Invalid response: {e}"}

class GoalSource(Enum):
    ALERT = "Alert (SMS/Email/Call)"
    HISTORICAL_AI = "Historical AI"
    KNOWLEDGE_AI = "Knowledge AI"

@dataclass
class AgentContext:
    last_alert_text: str = "" 
    last_goal_text: str = ""
    goal_source: GoalSource = GoalSource.ALERT
    iteration_count: int = 0
    
    cntr_no: List[str] = field(default_factory=list)
    container_id: List[str] = field(default_factory=list)
    correlation_id: List[str] = field(default_factory=list)
    message_ref: List[str] = field(default_factory=list)
    vessel_name_like: List[str] = field(default_factory=list)
    vessel_advice_no: List[str] = field(default_factory=list)
    vessel_id: List[str] = field(default_factory=list)
    error_hints: List[str] = field(default_factory=list)
    
    log_results: Dict[str, pd.DataFrame] = field(default_factory=dict)
    db_results: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    previous_log_queries: List[str] = field(default_factory=list)
    previous_db_queries: List[str] = field(default_factory=list)

    def merge(self, **kwargs):
        for k, vals in kwargs.items():
            seq = getattr(self, k, None)
            if not isinstance(seq, list):
                continue
            seen = set(seq)
            for v in vals:
                if v and v not in seen:
                    seq.append(v)
                    seen.add(v)

    def to_public(self) -> Dict:
        return {
            "iteration": self.iteration_count,
            "cntr_no": self.cntr_no,
            "container_id": self.container_id,
            "correlation_id": self.correlation_id,
            "message_ref": self.message_ref,
            "vessel_name_like": self.vessel_name_like,
            "vessel_advice_no": self.vessel_advice_no,
            "vessel_id": self.vessel_id,
            "error_hints": self.error_hints,
            "log_tables": list(self.log_results.keys()),
            "db_tables": list(self.db_results.keys()),
        }

@dataclass
class OutputAlert:
    Alert: str 
    Goal: str
    sql_logfile: Dict[str, str]
    sql_database: Dict[str, str]
    filtered_logfile: Dict[str, str]
    filtered_database: Dict[str, str]
    satisfy: bool
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "Alert": self.Alert,
            "Goal": self.Goal,
            "sql_logfile": self.sql_logfile,
            "sql_database": self.sql_database,
            "filtered_logfile": self.filtered_logfile,
            "filtered_database": self.filtered_database,
            "satisfy": self.satisfy,
            "explanation": self.explanation,
        }

class AlertAgent:
    """LLM-powered agent for root cause investigation"""
    
    MAX_ITERATIONS = 10
    def __init__(self):
        self.ctx = AgentContext()
        self._last_log_sql: Dict[str, str] = {}
        self._last_db_sql: Dict[str, str] = {}

    def ctx_to_public_dict(self) -> Dict:
        return self.ctx.to_public()

    # ----- Step 1: Plan from goal and generate LOG SQL -----
    def plan_from_goal(self, source: GoalSource, text) -> Dict:
        self.ctx.goal_source = source
    
        # Parse input based on source type
        if source is GoalSource.ALERT:
            # For ALERT: both alert and goal are the same
            self.ctx.last_alert_text = text or ""
            self.ctx.last_goal_text = text or ""
            # Extract initial candidates
            cand = extract_candidates(text)
            self.ctx.merge(**cand)
    
        elif source in [GoalSource.HISTORICAL_AI, GoalSource.KNOWLEDGE_AI]:
            try:
                self.ctx.last_alert_text = text.get("Alert", "")
                print(self.ctx.last_alert_text)
                
                # Keep Details as-is (the LLM will parse it)
                self.ctx.last_goal_text = text.get("Details", "")
                
                # Extract container number from Alert for quick reference
                cand = extract_candidates(self.ctx.last_goal_text)
                self.ctx.merge(**cand)
                
            except json.JSONDecodeError:
                # Fallback: treat as plain text
                self.ctx.last_alert_text = text or ""
                self.ctx.last_goal_text = text or ""
                cand = extract_candidates(text)
                self.ctx.merge(**cand)

        # Generate LOG SQL using LLM
        sql_pack = self.build_sql_for_logs()
        return {"sql_logfile": sql_pack, "context": self.ctx.to_public()}

    def build_sql_for_logs(self) -> Dict[str, str]:
        """Use LLM to generate SQL queries for log files"""
        
        self.ctx.iteration_count += 1
    
        log_summaries = {}
        for name, df in self.ctx.log_results.items():
            # Convert timestamps and dates to strings for JSON serialization
            sample_data = df.head(3).copy()
            for col in sample_data.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_data[col]) or \
                   pd.api.types.is_object_dtype(sample_data[col]):
                    sample_data[col] = sample_data[col].astype(str)
            
            log_summaries[name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "sample": sample_data.to_dict('records') if len(df) > 0 else []
            }
        
        previous_context = ""
        if self.ctx.iteration_count > 0:
            previous_context = f"""
Previous iteration findings:
- Log tables already queried: {list(self.ctx.log_results.keys())}
- Previous log queries: {json.dumps(self.ctx.previous_log_queries, indent=2)}

You should build upon these results. Include previous queries if they're still relevant, 
plus new queries based on identifiers found: {self.ctx.to_public()}
"""
        
        # ADD: Show available log tables and their schemas
        available_logs = "\n".join([
            f"- {name}: {', '.join(schema)}" 
            for name, schema in LOG_SCHEMAS.items()
        ])
        
        system_prompt = f"""You are a SQL query generator for log file investigation.

Original Alert: {self.ctx.last_alert_text}
Investigation Goal: {self.ctx.last_goal_text}
Goal Source: {self.ctx.goal_source.value}
- Iteration: {self.ctx.iteration_count + 1}/{self.MAX_ITERATIONS}
- Container numbers: {self.ctx.cntr_no}
- Container IDs: {self.ctx.container_id}
- Correlation IDs: {self.ctx.correlation_id}
- Message refs: {self.ctx.message_ref}
- Vessel names: {self.ctx.vessel_name_like}
- Error hints: {self.ctx.error_hints}

{previous_context}

AVAILABLE LOG FILES AND THEIR SCHEMAS:
{available_logs}

Log data already collected:
{json.dumps(log_summaries, indent=2)}

Generate SELECT queries for relevant log files. Return ONLY a JSON object with this structure:
{{
  "log_file_name": "SELECT * FROM log_file_name WHERE condition1 AND condition2 ORDER BY log_timestamp DESC LIMIT 5000",
  ...
}}

IMPORTANT Rules:
1. **ONLY query tables from the AVAILABLE LOG FILES list above**
2. **ONLY use columns that exist in the schema for each table**
3. Include BOTH previous queries (if still relevant) AND new queries based on discovered identifiers
4. Use newly discovered correlation_id, cntr_no, message_ref values to expand the search
5. Use LIKE for text searches in details column: LOWER(details) LIKE LOWER('%value%')
6. Use IN for exact matches: column IN ('val1', 'val2')
7. Always add ORDER BY log_timestamp DESC LIMIT 5000
8. Return ONLY the JSON object, no explanations
9. DO NOT invent table names - only use tables from the list above"""

        user_prompt = "Generate SQL queries for the relevant log files based on the current context and previous findings."
        
        response = _post_to_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        if "error" in response:
            return {"error_log": f"-- LLM Error: {response['error']}"}
        
        try:
            sql_pack = json.loads(response["content"])
            self._last_log_sql = sql_pack
            # CHANGED: Store queries for next iteration
            self.ctx.previous_log_queries = list(sql_pack.values())
            return sql_pack
        except Exception as e:
            return {"error_log": f"-- Failed to parse LLM response: {e}"}

    # ----- Step 2: Ingest LOG results -----
    def ingest_log_results(self, dfs: List[pd.DataFrame]):
        for df in dfs:
            key = self._guess_log_table(df)
            if not key:
                key = f"log_table_{len(self.ctx.log_results)+1}"
            # CHANGED: Update existing or add new
            self.ctx.log_results[key] = df
        
        self._mine_identifiers_from_logs()

    def _guess_log_table(self, df: pd.DataFrame) -> Optional[str]:
        cols = set([c.lower() for c in df.columns])
        for name, schema in LOG_SCHEMAS.items():
            sch = set([c.lower() for c in schema])
            if len(cols.intersection(sch)) >= max(3, len(sch)//2):
                return name
        return None

    def _mine_identifiers_from_logs(self):
        cntr, corr, msgref = [], [], []
        for df in self.ctx.log_results.values():
            for col in df.columns:
                cl = col.lower()
                if cl in ("cntr_no",):
                    cntr.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
                if cl in ("correlation_id", "corrid"):
                    corr.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
                if cl in ("message_ref",):
                    msgref.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
        self.ctx.merge(cntr_no=cntr, correlation_id=corr, message_ref=msgref)

    # ----- Step 3: Generate DATABASE SQL -----
    def build_sql_for_databases(self) -> Dict[str, str]:
        """Use LLM to generate SQL queries for database tables"""
        
        # CHANGED: Include previous results and queries
        log_summaries = {}
        for name, df in self.ctx.log_results.items():
            # Convert all non-primitive types to strings for JSON serialization
            sample_data = df.head(5).copy()
            for col in sample_data.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_data[col]) or \
                   pd.api.types.is_object_dtype(sample_data[col]):
                    sample_data[col] = sample_data[col].astype(str)
            
            log_summaries[name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "sample": sample_data.to_dict('records') if len(df) > 0 else []
            }
        
        db_summaries = {}
        for name, df in self.ctx.db_results.items():
            # Convert all non-primitive types to strings for JSON serialization
            sample_data = df.head(3).copy()
            for col in sample_data.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_data[col]) or \
                   pd.api.types.is_object_dtype(sample_data[col]):
                    sample_data[col] = sample_data[col].astype(str)
            
            db_summaries[name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "sample": sample_data.to_dict('records') if len(df) > 0 else []
            }
        
        previous_context = ""
        if self.ctx.iteration_count > 0:
            previous_context = f"""
Previous iteration findings:
- Database tables already queried: {list(self.ctx.db_results.keys())}
- Previous database queries: {json.dumps(self.ctx.previous_db_queries, indent=2)}

You should build upon these results. Include previous queries if they're still relevant,
plus new queries based on identifiers found.
"""
        
        # ADD: Show available database tables and their schemas
        available_dbs = "\n".join([
            f"- {name}: {', '.join(schema)}" 
            for name, schema in DB_SCHEMAS.items()
        ])
        
        system_prompt = f"""You are a SQL query generator for database investigation.

Original Alert: {self.ctx.last_alert_text}
Investigation Goal: {self.ctx.last_goal_text}
Goal Source: {self.ctx.goal_source.value}
- Iteration: {self.ctx.iteration_count + 1}/{self.MAX_ITERATIONS}
- Container numbers: {self.ctx.cntr_no}
- Container IDs: {self.ctx.container_id}
- Correlation IDs: {self.ctx.correlation_id}
- Message refs: {self.ctx.message_ref}
- Vessel IDs: {self.ctx.vessel_id}
- Vessel names: {self.ctx.vessel_name_like}

{previous_context}

AVAILABLE DATABASE TABLES AND THEIR SCHEMAS:
{available_dbs}

Log files analyzed:
{json.dumps(log_summaries, indent=2)}

Database data already collected:
{json.dumps(db_summaries, indent=2)}

Generate SELECT queries for relevant database tables. Return ONLY a JSON object:
{{
  "table_name": "SELECT * FROM table_name WHERE condition1 AND condition2 ORDER BY created_at DESC LIMIT 5000",
  ...
}}

IMPORTANT Rules:
1. **ONLY query tables from the AVAILABLE DATABASE TABLES list above**
2. The details given is a solution we used in the past. table names might change slightly. 
Adapt the solution to suit the AVAILABLE DATABASE TABLES list above.
3. **ONLY use columns that exist in the schema for each table**
4. Include BOTH previous queries (if still relevant) AND new queries based on discovered identifiers
5. Use newly discovered container_id, vessel_id, correlation_id, etc. to expand the search
6. Use IN for exact matches: column IN ('val1', 'val2')
7. Use LIKE for text searches: LOWER(column) LIKE LOWER('%value%')
8. Always add ORDER BY created_at DESC LIMIT 5000 (or appropriate timestamp column)
9. Return ONLY the JSON object, no explanations
10. DO NOT invent table names - only use tables from the list above"""

        user_prompt = "Generate SQL queries for the relevant database tables based on the logs, context, and previous findings."
        
        response = _post_to_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        if "error" in response:
            return {"error_db": f"-- LLM Error: {response['error']}"}
        
        try:
            sql_pack = json.loads(response["content"])
            self._last_db_sql = sql_pack
            # CHANGED: Store queries for next iteration
            self.ctx.previous_db_queries = list(sql_pack.values())
            return sql_pack
        except Exception as e:
            return {"error_db": f"-- Failed to parse LLM response: {e}"}

    # ----- Step 4: Ingest DATABASE results -----
    def ingest_db_results(self, dfs: List[pd.DataFrame]):
        for df in dfs:
            key = self._guess_db_table(df)
            if not key:
                key = f"db_table_{len(self.ctx.db_results)+1}"
            # CHANGED: Update existing or add new
            self.ctx.db_results[key] = df
        
        self._mine_identifiers_from_db()

    def _guess_db_table(self, df: pd.DataFrame) -> Optional[str]:
        cols = set([c.lower() for c in df.columns])
        for name, schema in DB_SCHEMAS.items():
            sch = set([c.lower() for c in schema])
            if len(cols.intersection(sch)) >= max(3, len(sch)//2):
                return name
        return None

    def _mine_identifiers_from_db(self):
        cntr, corr, msgref, cid, vid = [], [], [], [], []
        for df in self.ctx.db_results.values():
            for col in df.columns:
                cl = col.lower()
                if cl == "cntr_no":
                    cntr.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
                if cl == "correlation_id":
                    corr.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
                if cl == "message_ref":
                    msgref.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
                if cl == "container_id":
                    cid.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
                if cl == "vessel_id":
                    vid.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
        self.ctx.merge(cntr_no=cntr, correlation_id=corr, message_ref=msgref, 
                      container_id=cid, vessel_id=vid)

    # ----- Step 5: Final Analysis -----
    def analyze(self) -> OutputAlert:
        """Use LLM to analyze all data and determine if goal is satisfied"""
        
        # Prepare data summaries
        log_summaries = {}
        for name, df in self.ctx.log_results.items():
            # Convert all non-primitive types to strings for JSON serialization
            sample_data = df.head(50).copy()
            for col in sample_data.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_data[col]) or \
                   pd.api.types.is_object_dtype(sample_data[col]):
                    sample_data[col] = sample_data[col].astype(str)
        
            log_summaries[name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "data": sample_data.to_dict('records')
            }
    
        db_summaries = {}
        for name, df in self.ctx.db_results.items():
            # Convert all non-primitive types to strings for JSON serialization
            sample_data = df.head(50).copy()
            for col in sample_data.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_data[col]) or \
                   pd.api.types.is_object_dtype(sample_data[col]):
                    sample_data[col] = sample_data[col].astype(str)
        
            db_summaries[name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "data": sample_data.to_dict('records')
            }
    
        system_prompt = f"""You are a root cause analysis expert for port operations.

Original Alert: {self.ctx.last_alert_text}
Investigation Goal: {self.ctx.last_goal_text}
Goal Source: {self.ctx.goal_source.value}
Current Iteration: {self.ctx.iteration_count}/{self.MAX_ITERATIONS}

Log Files Analyzed:
{json.dumps(log_summaries, indent=2)}

Database Tables Analyzed:
{json.dumps(db_summaries, indent=2)}

Analyze the data and determine if the investigation goal is satisfied.

Return ONLY a JSON object with this EXACT structure:
{{
  "satisfy": true/false,
  "explanation": {{
    "summary": "Brief 1-2 sentence summary of the solution",
    "findings": {{
      "root_cause": "Detailed explanation of the root cause",
      "impacted_entities": ["List of affected containers, messages, vessels, etc."],
      "key_identifiers": {{
        "correlation_id": ["list of relevant correlation IDs"],
        "message_ref": ["list of relevant message refs"],
        "container_id": ["list of affected containers"],
        "edi_id": ["list of relevant EDI IDs"],
        "other": ["any other relevant identifiers"]
      }}
    }},
    "steps": [
      {{
        "step": 1,
        "action": "Description of what to do",
        "sql": "SELECT * FROM table WHERE condition; -- If SQL action is needed, otherwise null",
        "reason": "Why this step is needed"
      }},
      {{
        "step": 2,
        "action": "Description of next action",
        "sql": null,
        "reason": "Why this step is needed"
      }}
    ],
    "warnings": [
      "Warning 1: What to be careful about",
      "Warning 2: Another potential issue to watch"
    ],
    "verification": {{
      "how_to_verify": "Detailed steps to verify the fix is complete",
      "queries_to_run": [
        "SELECT * FROM table WHERE condition; -- Query to check status"
      ],
      "expected_outcome": "What the result should look like when issue is resolved"
    }}
  }}
}}

Common issues to look for:
1. Duplicate containers: Same cntr_no with different container_id or conflicting data
2. EDI stuck messages: status='ERROR' and ack_at is NULL
3. API failures: High rate of non-2xx http_status codes
4. Timestamp anomalies: Out-of-order events, conflicting timestamps
5. Missing correlations: Events that should be linked but aren't

IMPORTANT:
- If satisfy=true: Provide clear root cause and remediation steps
- If satisfy=false: Explain what additional data is needed (which tables, what identifiers) and suggest next queries to run
- Always include SQL queries in the "steps" array if database actions are needed
- Always include verification queries
- Return ONLY the JSON object, no markdown formatting"""

        user_prompt = "Analyze the data and determine if the investigation goal is satisfied. Provide structured explanation with actionable steps."
        
        response = _post_to_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        if "error" in response:
            satisfy = False
            explanation = {
                "summary": f"LLM Analysis Error: {response['error']}",
                "findings": {
                    "root_cause": "Unable to analyze due to LLM error",
                    "impacted_entities": [],
                    "key_identifiers": {}
                },
                "steps": [],
                "warnings": ["LLM communication failed"],
                "verification": {
                    "how_to_verify": "N/A - Analysis failed",
                    "queries_to_run": [],
                    "expected_outcome": "N/A"
                }
            }
        else:
            try:
                result = json.loads(response["content"])
                satisfy = result.get("satisfy", False)
                explanation = result.get("explanation", {
                    "summary": "No explanation provided",
                    "findings": {"root_cause": "Unknown", "impacted_entities": [], "key_identifiers": {}},
                    "steps": [],
                    "warnings": [],
                    "verification": {"how_to_verify": "N/A", "queries_to_run": [], "expected_outcome": "N/A"}
                })
            except Exception as e:
                satisfy = False
                explanation = {
                    "summary": f"Failed to parse analysis: {e}",
                    "findings": {
                        "root_cause": "Unable to parse LLM response",
                        "impacted_entities": [],
                        "key_identifiers": {}
                    },
                    "steps": [],
                    "warnings": ["Response parsing failed"],
                    "verification": {
                        "how_to_verify": "N/A - Parsing failed",
                        "queries_to_run": [],
                        "expected_outcome": "N/A"
                    }
                }
    
        # Build output
        log_shapes = {k: f"DataFrame(rows={len(v)}, cols={len(v.columns)})" 
                     for k, v in self.ctx.log_results.items()}
        db_shapes = {k: f"DataFrame(rows={len(v)}, cols={len(v.columns)})" 
                    for k, v in self.ctx.db_results.items()}
    
        return OutputAlert(
            Alert=self.ctx.last_alert_text,
            Goal=self.ctx.last_goal_text,
            sql_logfile={k: dedent_sql(v) for k, v in self._last_log_sql.items()},
            sql_database={k: dedent_sql(v) for k, v in self._last_db_sql.items()},
            filtered_logfile=log_shapes,
            filtered_database=db_shapes,
            satisfy=satisfy,
            explanation=json.dumps(explanation, indent=2),
        )