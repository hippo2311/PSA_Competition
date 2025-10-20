"""
main.py
Streamlit UI for LLM-powered AlertAgent workflow with real-time log ingestion
"""
import streamlit as st
import datetime
import json
import re
from email.utils import parseaddr  # ✅ ensure "To" shows only the email address

from runsql import run_query
from ApplicationLog.ingest import start_ingestion_background
from Agents.AlertAgent import (
    AlertAgent,
    GoalSource,
    OutputAlert,
)
from Agents.ExtractorAgent import ExtractorAgent, DEFAULT_CONTACTS_PATH

from Embeddings.knowledge_agent import *
from Embeddings.historical_agent import *
from utils import mark_ticket_resolved, get_case_logs_sheet_url
from send_email import send_email_to_stakeholder


# -----------------------------------
# 🌐 Streamlit Config
# -----------------------------------
st.set_page_config(
    page_title="PSA AlertAgent — Root Cause Investigator",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ PSA AlertAgent — Root Cause Investigator")
st.caption("Real-time Logs → MySQL → LLM SQL → Analysis")

# -----------------------------------
# 🚀 Start Background Log Ingestion
# -----------------------------------
if "ingestion_thread" not in st.session_state:
    thread = start_ingestion_background(sleep_interval=60)

# -----------------------------------
# 🧠 Initialize LLM Agents (one per model)
# -----------------------------------
if "agent_alert" not in st.session_state:
    st.session_state.agent_alert = AlertAgent()
if "agent_historical" not in st.session_state:
    st.session_state.agent_historical = AlertAgent()
if "agent_knowledge" not in st.session_state:
    st.session_state.agent_knowledge = AlertAgent()

if "log_results_alert" not in st.session_state:
    st.session_state.log_results_alert = {}
if "db_results_alert" not in st.session_state:
    st.session_state.db_results_alert = {}

if "log_results_historical" not in st.session_state:
    st.session_state.log_results_historical = {}
if "db_results_historical" not in st.session_state:
    st.session_state.db_results_historical = {}

if "log_results_knowledge" not in st.session_state:
    st.session_state.log_results_knowledge = {}
if "db_results_knowledge" not in st.session_state:
    st.session_state.db_results_knowledge = {}

if "workflow_completed" not in st.session_state:
    st.session_state.workflow_completed = False

# --- Context display ---
with st.expander("⚙️ Current Context (keywords & history)", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Alert Agent:**")
        st.json(st.session_state.agent_alert.ctx_to_public_dict(), expanded=False)

    with col2:
        st.markdown("**Historical Agent:**")
        st.json(st.session_state.agent_historical.ctx_to_public_dict(), expanded=False)

    with col3:
        st.markdown("**Knowledge Agent:**")
        st.json(st.session_state.agent_knowledge.ctx_to_public_dict(), expanded=False)

# -----------------------------------
# Step 1: Provide Goal
# -----------------------------------
st.markdown("---")
st.subheader("Step 1: Provide Alert")

goal_text = st.text_area(
    "Paste alert text",
    height=160,
    placeholder=(
        "Example:\n"
        "RE: Email ALR-861600 | CMAU0000020 - Duplicate Container information received\n"
        "Please check container CMAU0000020. Customer seeing 2 identical containers."
    ),
)

# -----------------------------------
# Three-Column Investigation
# -----------------------------------
st.markdown("---")

col_alert, col_historical, col_knowledge = st.columns(3)

# ========== COLUMN 1: ALERT MODEL ==========
with col_alert:
    st.markdown("### 🚨 Alert Model")

    if st.button("🚀 Run Alert Investigation", type="primary", key="btn_alert", use_container_width=True):
        st.session_state.log_results_alert = {}
        st.session_state.db_results_alert = {}

        agent_alert = st.session_state.agent_alert

        with st.spinner("🧭 Generating LOG SQL..."):
            plan = agent_alert.plan_from_goal(GoalSource.ALERT, goal_text)
            sql_logs = plan.get("sql_logfile", {})
            st.session_state["last_sql_logs_alert"] = sql_logs

        if sql_logs:
            with st.spinner(f"🧭 Running {len(sql_logs)} LOG queries..."):
                for name, sql in sql_logs.items():
                    df = run_query(sql)
                    if not df.empty:
                        st.session_state.log_results_alert[name] = df
                        agent_alert.ingest_log_results([df])

        with st.spinner("🏗️ Generating DATABASE SQL..."):
            sql_db = agent_alert.build_sql_for_databases()
            st.session_state["last_sql_db_alert"] = sql_db

        if sql_db:
            with st.spinner(f"🏗️ Running {len(sql_db)} DATABASE queries..."):
                for name, sql in sql_db.items():
                    df = run_query(sql)
                    if not df.empty:
                        st.session_state.db_results_alert[name] = df
                        agent_alert.ingest_db_results([df])

        with st.spinner("🧪 Analyzing..."):
            out: OutputAlert = agent_alert.analyze()
            st.session_state["analysis_output_alert"] = out

        st.success("✅ Alert Investigation Complete!")
        st.rerun()

    # Display Alert results
    if "analysis_output_alert" in st.session_state:
        with st.expander("📊 Alert Results", expanded=True):
            out = st.session_state["analysis_output_alert"]
            explanation = json.loads(out.explanation)

            st.markdown(f"**Summary:** {explanation.get('summary', 'N/A')}")
            st.markdown(f"**Root Cause:** {explanation.get('findings', {}).get('root_cause', 'N/A')}")

            if out.satisfy:
                st.success("✅ Goal SATISFIED")
            else:
                st.warning("⚠️ Goal NOT SATISFIED")

            complete_json = json.dumps(out.to_dict(), indent=2)
            st.download_button(
                "📥 Download Report",
                data=complete_json,
                file_name=f"alert_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_alert"
            )

# ========== COLUMN 2: HISTORICAL MODEL ==========
with col_historical:
    st.markdown("### 📚 Historical Model")

    if st.button("🚀 Run Historical Investigation", type="primary", key="btn_historical", use_container_width=True):
        st.session_state.log_results_historical = {}
        st.session_state.db_results_historical = {}

        agent_historical = st.session_state.agent_historical

        with st.spinner("🔍 Querying historical model..."):
            goal_text_historical = historical_model(goal_text)

        with st.spinner("🧭 Generating LOG SQL..."):
            plan = agent_historical.plan_from_goal(GoalSource.HISTORICAL_AI, goal_text_historical)
            sql_logs = plan.get("sql_logfile", {})
            st.session_state["last_sql_logs_historical"] = sql_logs

        if sql_logs:
            with st.spinner(f"🧭 Running {len(sql_logs)} LOG queries..."):
                for name, sql in sql_logs.items():
                    df = run_query(sql)
                    if not df.empty:
                        st.session_state.log_results_historical[name] = df
                        agent_historical.ingest_log_results([df])

        with st.spinner("🏗️ Generating DATABASE SQL..."):
            sql_db = agent_historical.build_sql_for_databases()
            st.session_state["last_sql_db_historical"] = sql_db

        if sql_db:
            with st.spinner(f"🏗️ Running {len(sql_db)} DATABASE queries..."):
                for name, sql in sql_db.items():
                    df = run_query(sql)
                    if not df.empty:
                        st.session_state.db_results_historical[name] = df
                        agent_historical.ingest_db_results([df])

        with st.spinner("🧪 Analyzing..."):
            out: OutputAlert = agent_historical.analyze()
            st.session_state["analysis_output_historical"] = out

        st.success("✅ Historical Investigation Complete!")
        st.rerun()

    # Display Historical results
    if "analysis_output_historical" in st.session_state:
        with st.expander("📊 Historical Results", expanded=True):
            out = st.session_state["analysis_output_historical"]
            explanation = json.loads(out.explanation)

            st.markdown(f"**Summary:** {explanation.get('summary', 'N/A')}")
            st.markdown(f"**Root Cause:** {explanation.get('findings', {}).get('root_cause', 'N/A')}")

            if out.satisfy:
                st.success("✅ Goal SATISFIED")
            else:
                st.warning("⚠️ Goal NOT SATISFIED")

            # Download button
            complete_json = json.dumps(out.to_dict(), indent=2)
            st.download_button(
                "📥 Download Report",
                data=complete_json,
                file_name=f"historical_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_historical"
            )

# ========== COLUMN 3: KNOWLEDGE MODEL ==========
with col_knowledge:
    st.markdown("### 🧠 Knowledge Model")

    if st.button("🚀 Run Knowledge Investigation", type="primary", key="btn_knowledge", use_container_width=True):
        st.session_state.log_results_knowledge = {}
        st.session_state.db_results_knowledge = {}

        agent_knowledge = st.session_state.agent_knowledge

        with st.spinner("🔍 Querying knowledge base..."):
            goal_text_knowledge = knowledge_model(goal_text)

        with st.spinner("🧭 Generating LOG SQL..."):
            plan = agent_knowledge.plan_from_goal(GoalSource.KNOWLEDGE_AI, goal_text_knowledge)
            sql_logs = plan.get("sql_logfile", {})
            st.session_state["last_sql_logs_knowledge"] = sql_logs

        if sql_logs:
            with st.spinner(f"🧭 Running {len(sql_logs)} LOG queries..."):
                for name, sql in sql_logs.items():
                    df = run_query(sql)
                    if not df.empty:
                        st.session_state.log_results_knowledge[name] = df
                        agent_knowledge.ingest_log_results([df])

        with st.spinner("🏗️ Generating DATABASE SQL..."):
            sql_db = agent_knowledge.build_sql_for_databases()
            st.session_state["last_sql_db_knowledge"] = sql_db

        if sql_db:
            with st.spinner(f"🏗️ Running {len(sql_db)} DATABASE queries..."):
                for name, sql in sql_db.items():
                    df = run_query(sql)
                    if not df.empty:
                        st.session_state.db_results_knowledge[name] = df
                        agent_knowledge.ingest_db_results([df])

        with st.spinner("🧪 Analyzing..."):
            out: OutputAlert = agent_knowledge.analyze()
            st.session_state["analysis_output_knowledge"] = out

        st.success("✅ Knowledge Investigation Complete!")
        st.rerun()

    # Display Knowledge results
    if "analysis_output_knowledge" in st.session_state:
        with st.expander("📊 Knowledge Results", expanded=True):
            out = st.session_state["analysis_output_knowledge"]
            explanation = json.loads(out.explanation)

            st.markdown(f"**Summary:** {explanation.get('summary', 'N/A')}")
            st.markdown(f"**Root Cause:** {explanation.get('findings', {}).get('root_cause', 'N/A')}")

            if out.satisfy:
                st.success("✅ Goal SATISFIED")
            else:
                st.warning("⚠️ Goal NOT SATISFIED")

            # Download button
            complete_json = json.dumps(out.to_dict(), indent=2)
            st.download_button(
                "📥 Download Report",
                data=complete_json,
                file_name=f"knowledge_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_knowledge"
            )

# -----------------------------------
# Detailed Results Section
# -----------------------------------
st.markdown("---")
st.subheader("📊 Detailed Investigation Results")

tab_alert, tab_historical, tab_knowledge = st.tabs(["🚨 Alert Model", "📚 Historical Model", "🧠 Knowledge Model"])

# Helper function to display detailed results
def display_detailed_results(tab, model_name, log_results, db_results, last_sql_logs_key, last_sql_db_key, analysis_key):
    with tab:
        # LOG Query Results
        with st.expander("🧭 LOG Query Results", expanded=False):
            if log_results:
                for name, df in log_results.items():
                    st.markdown(f"### 🔎 {name} ({len(df)} rows)")
                    if last_sql_logs_key in st.session_state and name in st.session_state[last_sql_logs_key]:
                        st.code(st.session_state[last_sql_logs_key][name], language="sql")
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download {name}.csv",
                        data=csv,
                        file_name=f"{name}_{model_name}.csv",
                        mime="text/csv",
                        key=f"download_log_{model_name}_{name}"
                    )
                    st.markdown("---")
            else:
                st.info("No LOG results available")

        # DATABASE Query Results
        with st.expander("🏗️ DATABASE Query Results", expanded=False):
            if db_results:
                for name, df in db_results.items():
                    st.markdown(f"### 🗄️ {name} ({len(df)} rows)")
                    if last_sql_db_key in st.session_state and name in st.session_state[last_sql_db_key]:
                        st.code(st.session_state[last_sql_db_key][name], language="sql")
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download {name}.csv",
                        data=csv,
                        file_name=f"{name}_{model_name}.csv",
                        mime="text/csv",
                        key=f"download_db_{model_name}_{name}"
                    )
                    st.markdown("---")
            else:
                st.info("No DATABASE results available")

        # Full Analysis
        with st.expander("🧪 Complete Analysis & Root Cause", expanded=True):
            if analysis_key in st.session_state:
                out = st.session_state[analysis_key]
                explanation = json.loads(out.explanation)

                complete_output = {
                    "Alert": out.Alert,
                    "Goal": out.Goal,
                    "sql_logfile": out.sql_logfile,
                    "sql_database": out.sql_database,
                    "filtered_logfile": out.filtered_logfile,
                    "filtered_database": out.filtered_database,
                    "satisfy": out.satisfy,
                    "explanation": explanation
                }

                st.json(complete_output, expanded=True)

                st.markdown("---")
                if out.satisfy:
                    st.success("✅ Investigation Goal SATISFIED")
                else:
                    st.warning("⚠️ Investigation Goal NOT SATISFIED")

                # Structured display
                st.markdown("### 🔍 Detailed Analysis")
                st.markdown("#### Summary")
                st.info(explanation.get("summary", "No summary provided"))

                st.markdown("#### Findings")
                findings = explanation.get("findings", {})
                st.markdown(f"**Root Cause:** {findings.get('root_cause', 'Unknown')}")

                if findings.get("impacted_entities"):
                    st.markdown("**Impacted Entities:**")
                    for entity in findings["impacted_entities"]:
                        st.markdown(f"- {entity}")

                if findings.get("key_identifiers"):
                    st.markdown("**Key Identifiers:**")
                    st.json(findings["key_identifiers"])

                st.markdown("#### Remediation Steps")
                steps = explanation.get("steps", [])
                for step_obj in steps:
                    step_num = step_obj.get("step", "?")
                    action = step_obj.get("action", "No action")
                    sql = step_obj.get("sql")
                    reason = step_obj.get("reason", "")

                    st.markdown(f"**Step {step_num}:** {action}")
                    if reason:
                        st.markdown(f"*Reason:* {reason}")
                    if sql and sql.strip() and sql.strip().lower() != "null":
                        st.code(sql, language="sql")
                    st.markdown("---")

                warnings = explanation.get("warnings", [])
                if warnings:
                    st.markdown("#### ⚠️ Warnings")
                    for warning in warnings:
                        st.warning(warning)

                st.markdown("#### ✅ Verification")
                verification = explanation.get("verification", {})
                st.markdown(f"**How to Verify:** {verification.get('how_to_verify', 'N/A')}")

                queries = verification.get("queries_to_run", [])
                if queries:
                    st.markdown("**Verification Queries:**")
                    for i, query in enumerate(queries, 1):
                        st.code(query, language="sql")

                st.markdown(f"**Expected Outcome:** {verification.get('expected_outcome', 'N/A')}")
            else:
                st.info(f"No analysis available for {model_name} yet. Run the investigation first.")

# Display detailed results for each model
display_detailed_results(
    tab_alert, "alert",
    st.session_state.log_results_alert,
    st.session_state.db_results_alert,
    "last_sql_logs_alert",
    "last_sql_db_alert",
    "analysis_output_alert"
)

display_detailed_results(
    tab_historical, "historical",
    st.session_state.log_results_historical,
    st.session_state.db_results_historical,
    "last_sql_logs_historical",
    "last_sql_db_historical",
    "analysis_output_historical"
)

display_detailed_results(
    tab_knowledge, "knowledge",
    st.session_state.log_results_knowledge,
    st.session_state.db_results_knowledge,
    "last_sql_logs_knowledge",
    "last_sql_db_knowledge",
    "analysis_output_knowledge"
)

# -----------------------------------
# Executor AI Section (after investigations)
# -----------------------------------
st.markdown("---")
st.header("📋 Executor AI — Generate Ticket & Email")
st.caption("Use the investigation results to generate a structured ticket and notification email")

# Check if at least one investigation has completed
has_alert = "analysis_output_alert" in st.session_state
has_historical = "analysis_output_historical" in st.session_state
has_knowledge = "analysis_output_knowledge" in st.session_state

if not (has_alert or has_historical or has_knowledge):
    st.info("ℹ️ Run at least one investigation above to use the Executor AI")
else:
    # Import ExtractorAgent
    if "extractor_agent" not in st.session_state:
        st.session_state.extractor_agent = ExtractorAgent()

    extractor_agent = st.session_state.extractor_agent

    st.caption(
        f"Contacts: default PDF is `{DEFAULT_CONTACTS_PATH}`. "
        "You can paste the whole contacts text below to override."
    )

    # Prepare complete outputs from investigation results
    col1, col2, col3 = st.columns(3)

    # Initialize variables
    hist_input = None
    kb_input = None
    judge_input = None
    use_historical = False
    use_knowledge = False
    use_alert = False

    with col1:
        st.markdown("#### 📚 Historical AI Complete Output")
        if has_historical:
            hist_out = st.session_state["analysis_output_historical"]
            explanation = json.loads(hist_out.explanation)

            # Create complete output matching the format
            hist_complete = {
                "Alert": hist_out.Alert,
                "Goal": hist_out.Goal,
                "sql_logfile": hist_out.sql_logfile,
                "sql_database": hist_out.sql_database,
                "filtered_logfile": hist_out.filtered_logfile,
                "filtered_database": hist_out.filtered_database,
                "satisfy": hist_out.satisfy,
                "explanation": explanation
            }

            # Display complete output
            with st.expander("View Complete Output", expanded=False):
                st.json(hist_complete, expanded=False)

            # Format for ExtractorAgent
            hist_input = {
                "alert_message": hist_complete["Alert"],
                "goal": hist_complete["Goal"],
                "explanation": hist_complete["explanation"],
            }

            st.markdown("**Formatted for Executor:**")
            st.json(hist_input, expanded=False)
            use_historical = st.checkbox("✅ Use Historical AI output", value=True, key="use_hist")
        else:
            st.info("No Historical results available")

    with col2:
        st.markdown("#### 🧠 Knowledge AI Complete Output")
        if has_knowledge:
            kb_out = st.session_state["analysis_output_knowledge"]
            explanation = json.loads(kb_out.explanation)

            # Create complete output matching the format
            kb_complete = {
                "Alert": kb_out.Alert,
                "Goal": kb_out.Goal,
                "sql_logfile": kb_out.sql_logfile,
                "sql_database": kb_out.sql_database,
                "filtered_logfile": kb_out.filtered_logfile,
                "filtered_database": kb_out.filtered_database,
                "satisfy": kb_out.satisfy,
                "explanation": explanation
            }

            # Display complete output
            with st.expander("View Complete Output", expanded=False):
                st.json(kb_complete, expanded=False)

            # Format for ExtractorAgent
            kb_input = {
                "alert_message": kb_complete["Alert"],
                "goal": kb_complete["Goal"],
                "explanation": kb_complete["explanation"],
            }

            st.markdown("**Formatted for Executor:**")
            st.json(kb_input, expanded=False)
            use_knowledge = st.checkbox("✅ Use Knowledge AI output", value=True, key="use_kb")
        else:
            st.info("No Knowledge results available")

    with col3:
        st.markdown("#### 🚨 Alert AI Complete Output")
        if has_alert:
            alert_out = st.session_state["analysis_output_alert"]
            explanation = json.loads(alert_out.explanation)

            # Create complete output matching the format
            alert_complete = {
                "Alert": alert_out.Alert,
                "Goal": alert_out.Goal,
                "sql_logfile": alert_out.sql_logfile,
                "sql_database": alert_out.sql_database,
                "filtered_logfile": alert_out.filtered_logfile,
                "filtered_database": alert_out.filtered_database,
                "satisfy": alert_out.satisfy,
                "explanation": explanation
            }

            # Display complete output
            with st.expander("View Complete Output", expanded=False):
                st.json(alert_complete, expanded=False)

            # Format as judge output text
            judge_input = {
                "alert_message": alert_complete["Alert"],
                "goal": alert_complete["Goal"],
                "explanation": alert_complete["explanation"],
            }

            st.markdown("**Formatted for Executor:**")
            st.json(judge_input, expanded=False)
            use_alert = st.checkbox("✅ Use Alert AI output", value=True, key="use_alert")
        else:
            st.info("No Alert results available")

    # Selection Summary
    st.markdown("---")
    st.subheader("📋 Selected Inputs for Executor AI")

    selected_count = sum([use_historical, use_knowledge, use_alert])
    if selected_count == 0:
        st.warning("⚠️ No AI outputs selected. Please check at least one checkbox above.")
    else:
        st.success(f"✅ {selected_count} AI output(s) selected")

        selected_list = []
        if use_historical:
            selected_list.append("📚 Historical AI")
        if use_knowledge:
            selected_list.append("🧠 Knowledge AI")
        if use_alert:
            selected_list.append("🚨 Alert AI")

        st.markdown("**Will use:** " + ", ".join(selected_list))

    # Contacts override
    st.markdown("---")
    st.subheader("📞 Contacts Override (optional)")
    use_contacts_override = st.checkbox("Override contacts by pasting full text")
    contacts_text = None
    if use_contacts_override:
        contacts_text = st.text_area(
            "Paste the full 'Product Team Escalation Contacts' text",
            placeholder="Name: ...\nRole: ...\nEmail: ...\nPhone: ...\nScenarios: advice, uniqueness, database\n...",
            height=200
        )

    # Ticket prefix
    ticket_prefix = st.text_input("Ticket prefix (auto-ID)", value="TCK")

    # Timestamp helper
    def _ensure_alert_start_ts_iso() -> str:
        if "alert_start_time" not in st.session_state:
            from zoneinfo import ZoneInfo
            st.session_state["alert_start_time"] = datetime.datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
        return st.session_state["alert_start_time"]

    # Run Executor AI
    st.markdown("---")
    run_executor = st.button("🚀 Generate Ticket & Email", type="primary", use_container_width=True, disabled=(selected_count == 0))

    if run_executor:
        # Prepare inputs based on checkboxes
        final_hist = hist_input if use_historical else None
        final_kb = kb_input if use_knowledge else None
        final_judge = judge_input if use_alert else None

        with st.spinner("📝 Generating ticket and email..."):
            alert_start_ts_iso = _ensure_alert_start_ts_iso()

            final = extractor_agent.run(
                historical_data=final_hist,
                knowledge_base=final_kb,
                judge_output=final_judge,
                contacts_pdf=None,
                contacts_text=contacts_text,
                return_dict=True,
                alert_start_ts=alert_start_ts_iso,
                ticket_prefix=ticket_prefix or "TCK",
            )

            st.session_state["executor_result"] = final

        st.success("✅ Ticket & Email Generated!")
        st.rerun()

    # Display Executor Results
    if "executor_result" in st.session_state:
        final = st.session_state["executor_result"]

        st.markdown("---")
        st.subheader("📊 Executor AI Results")

        # Render markdown output
        try:
            rendered_md = extractor_agent._fmt_text_output_markdown(final)
        except Exception:
            rendered_md = (
                f"**Executive Summary**\n\n"
                f"- **Alert message:** {final.get('_alert_message','')}\n"
                f"- **Problem:** {final.get('_problem','')}\n"
                f"- **RCA:** {final.get('_rca','')}\n\n"
                f"_Ticket created: {final.get('_ticket_id','(unknown)')}_"
            )

        with st.expander("📄 Full Report", expanded=True):
            st.markdown(rendered_md)

        # Ticket info
        ticket_id = final.get("_ticket_id")
        if ticket_id:
            st.success(f"✅ Ticket created: **{ticket_id}**")
        else:
            st.info("ℹ️ Ticket ID missing in output (CSV append may have failed).")

        import os
        csv_path = final.get("_ticket_csv") or os.path.join("Outputs", "tickets_log.csv")
        st.caption(f"Ticket CSV: `{csv_path}`")
        if final.get("_ticket_error"):
            st.error(final["_ticket_error"])

        # -----------------------------------
        # 📧 Email preview (To = email only, no display name)
        # -----------------------------------
        st.markdown("---")
        st.subheader("📧 Ready-to-Send Email")

        notify = (final.get("_notify_email") or {}).copy()

        # Fallback: if ExtractorAgent didn't produce _notify_email, use escalation.email
        if not notify:
            esc_email = (final.get("escalation") or {}).get("email") or ""
            # Build a simple subject/body using available fields
            default_subject = f"[{final.get('_ticket_id','INC')}] {final.get('_problem') or 'Incident update'}"
            default_body = (final.get("_best_solution_summary") or "").strip()
            notify = {"to": esc_email, "subject": default_subject, "body": default_body}

        # Normalize To → keep only pure email, strip "Name <email@...>"
        to_raw = (notify.get("to") or "").strip()
        _, to_addr = parseaddr(to_raw)
        to_addr = to_addr or to_raw  # keep raw if parse fails
        notify["to"] = to_addr

        if notify.get("to"):
            col_email1, col_email2 = st.columns([3, 1])

            with col_email1:
                st.write(f"**To:** {notify['to']}")  # only email
                st.write(f"**Subject:** {notify.get('subject','')}")
                st.code((notify.get("body","") or "").strip(), language="text")

            with col_email2:
                # Download .eml
                eml = f"To: {notify['to']}\nSubject: {notify.get('subject','')}\n\n{notify.get('body','')}"
                eml_bytes = eml.encode("utf-8")
                eml_name = f"{ticket_id or 'incident'}_email.eml"
                st.download_button(
                    "⬇️ Download .eml",
                    data=eml_bytes,
                    file_name=eml_name,
                    mime="message/rfc822",
                    use_container_width=True
                )
        else:
            st.info("No email recipient available.")

        # Optional: send email via your backend helper
        if st.button("Send email to relevant stakeholders"):
            send_email_to_stakeholder(notify, final)

        # Download generated PDFs
        st.markdown("---")
        st.subheader("📥 Download Generated Files")

        def offer_download_if_exists(label: str, path: str):
            import os
            if not path:
                return
            if not os.path.exists(path):
                st.write(f"**{label}:** {path} *(file not found)*")
                return
            with open(path, "rb") as f:
                data = f.read()
            st.download_button(
                f"⬇️ {label}",
                data=data,
                file_name=os.path.basename(path),
                use_container_width=True
            )

        col_pdf1, col_pdf2, col_pdf3 = st.columns(3)

        with col_pdf1:
            offer_download_if_exists("📚 Historical Solution PDF", final.get("historical_data_ai_solution_pdf"))

        with col_pdf2:
            offer_download_if_exists("🧠 Knowledge Solution PDF", final.get("knowledge_base_ai_solution_pdf"))

        with col_pdf3:
            offer_download_if_exists("🚨 Alert Solution PDF", final.get("our_own_ai_solution_pdf"))

        # ========== NEW: Google Sheets Integration ==========
        st.markdown("---")
        st.subheader("📊 Ticket Management")

        col_sheets1, col_sheets2 = st.columns(2)

        with col_sheets1:
            st.markdown("#### 📋 View Tickets")
            from utils import get_tickets_sheet_url

            tickets_url = get_tickets_sheet_url()
            st.link_button(
                "🔗 Open Tickets Sheet",
                tickets_url,
                use_container_width=True
            )

            st.caption("All generated tickets are logged here")

        with col_sheets2:
            st.markdown("#### ✅ Mark as Resolved")

            if st.button("✅ Ticket Resolved - Move to Case Logs",
                         type="secondary",
                         use_container_width=True):

                with st.spinner("📝 Moving ticket to Case Logs..."):
                    success = mark_ticket_resolved()

                if success:
                    st.success("✅ Ticket marked as resolved and copied to Case Logs!")
                    case_logs_url = get_case_logs_sheet_url()
                    st.link_button("🔗 View Case Logs", case_logs_url)
                else:
                    st.error("❌ Failed to resolve ticket. Check error messages above.")

st.link_button(
    "📊 View Looker Studio Dashboard",
    "https://lookerstudio.google.com/reporting/92fd24f6-779b-43ee-a9db-f374bbe811eb",
    type="primary",
    use_container_width=True
)

# -----------------------------------
# Reset Button
# -----------------------------------
st.markdown("---")
if st.button("🔄 Reset All", use_container_width=True):
    st.session_state.agent_alert = AlertAgent()
    st.session_state.agent_historical = AlertAgent()
    st.session_state.agent_knowledge = AlertAgent()

    st.session_state.log_results_alert = {}
    st.session_state.db_results_alert = {}
    st.session_state.log_results_historical = {}
    st.session_state.db_results_historical = {}
    st.session_state.log_results_knowledge = {}
    st.session_state.db_results_knowledge = {}

    st.session_state.workflow_completed = False

    for key in list(st.session_state.keys()):
        if key.startswith("last_sql_") or key.startswith("analysis_output_") or key.startswith("executor_"):
            del st.session_state[key]

    st.rerun()
