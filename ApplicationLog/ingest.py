import os
import re
import time
import csv
import signal
import threading
from datetime import datetime
import mysql.connector
from mysql.connector import errorcode
import pandas as pd

# === CONFIG ===
LOG_DIR = "."
PRINT_DEBUG = True

DB_CONFIG = {
    "host": "psahackathon.ckn0s0ok2sbi.us-east-1.rds.amazonaws.com",
    "user": "admin",
    "password": "t0500224E",  # ⚠️ replace with your actual password
    "database": "appdb",
    "port": 3306
}

# === HELPERS ===
TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z?")
KV_RE = re.compile(r'(\w+)=(".*?"|\S+)')  # Handles quoted + unquoted pairs


def safe_dt(s):
    """Safely converts ISO timestamp string to datetime."""
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None


# === HEADERS ===
CSV_HEADERS = {
    "api_event_log": ["log_timestamp", "type", "details", "api_event_id", "container_id", "correlation_id"],
    "berth_application_log": ["log_timestamp", "type", "details", "system_vessel_name", "vessel_advice_no", "application_no", "correlation_id"],
    "container_service_log": ["log_timestamp", "type", "details", "cntr_no", "correlation_id"],
    "vessel_advice_log": ["log_timestamp", "type", "details", "vesselName", "system_vessel_name", "corrId"],
    "vessel_registry_log": ["log_timestamp", "type", "details", "vessel_id", "imo_no", "old_flag", "new_flag", "user"],
    "edi_advice_log": ["log_timestamp", "type", "details", "corrId", "messageType", "httpStatus", "code"]
}

csv_writers = {}
csv_files = {}


# === CSV WRITER ===
def write_csv_row(table, values):
    header_len = len(CSV_HEADERS[table])
    safe_values = ["NULL" if (v is None or v == "") else v for v in values]
    while len(safe_values) < header_len:
        safe_values.append("NULL")

    csv_path = f"{table}.csv"
    if table not in csv_writers:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS[table])
        csv_files[table] = open(csv_path, "a", newline="", encoding="utf-8")
        csv_writers[table] = csv.writer(csv_files[table])
        print(f"[INIT] Writing to {csv_path}")

    csv_writers[table].writerow(safe_values)
    csv_files[table].flush()


# === PARSER ===
def parse_line(service, line):
    """Parses one log line and returns (table_name, values) tuple."""
    result = None

    # ✅ Extract timestamp, log level, and details (everything after level)
    log_re = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[^\s]+)\s+(?P<level>INFO|DEBUG|ERROR|WARN)\s+(?P<details>.+)$"
    )

    m = log_re.match(line.strip())
    if not m:
        return None

    ts = safe_dt(m.group("timestamp"))
    level = m.group("level").upper()
    details = m.group("details").strip()

    # Normalize whitespace and truncate to 255 chars for DB unique keys
    details = re.sub(r"\s+", " ", details)[:255]

    # Extract key-value pairs for optional enrichment
    kv = {k: v.strip('"') for k, v in KV_RE.findall(details)}

    # === Assign correct table ===
    if service == "api_event_service.log":
        table = "api_event_log"
        values = (ts, level, details,
                  kv.get("api_event_id"),
                  kv.get("container_id"),
                  kv.get("correlation_id"))

    elif service == "berth_application_service.log":
        table = "berth_application_log"
        values = (ts, level, details,
                  kv.get("system_vessel_name"),
                  kv.get("vessel_advice_no"),
                  kv.get("application_no"),
                  kv.get("correlation_id"))

    elif service == "container_service.log":
        table = "container_service_log"
        values = (ts, level, details,
                  kv.get("cntr_no"),
                  kv.get("correlation_id"))

    elif service == "vessel_advice_service.log":
        table = "vessel_advice_log"
        values = (ts, level, details,
                  kv.get("vesselName") or kv.get("vesselname"),
                  kv.get("system_vessel_name") or kv.get("systemVesselName"),
                  kv.get("corrId") or kv.get("correlation_id"))

    elif service == "vessel_registry_service.log":
        table = "vessel_registry_log"
        values = (ts, level, details,
                  kv.get("vessel_id"),
                  kv.get("imo_no"),
                  kv.get("old_flag"),
                  kv.get("new_flag"),
                  kv.get("user"))

    elif service == "edi_advice_service.log":
        table = "edi_advice_log"
        values = (ts, level, details,
                  kv.get("corrId"),
                  kv.get("messageType"),
                  kv.get("httpStatus"),
                  kv.get("code"))

    else:
        return None

    return (table, values)


# === MYSQL PUSH FUNCTION ===
def push_csv_to_mysql():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("\n🚀 Pushing CSV data to MySQL...")

        for table_name in CSV_HEADERS.keys():
            csv_file = f"{table_name}.csv"
            file_path = os.path.join(LOG_DIR, csv_file)
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)

            if df.empty:
                print(f"⚠️ {table_name}: Empty CSV, skipping.")
                continue

            # ✅ Deduplicate before push
            if {"log_timestamp", "type", "details"} <= set(df.columns):
                df.drop_duplicates(subset=["log_timestamp", "type", "details"], inplace=True)
            else:
                df.drop_duplicates(inplace=True)

            columns = ", ".join(df.columns)
            placeholders = ", ".join(["%s"] * len(df.columns))

            insert_query = f"""
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE
                {", ".join([f"{col}=VALUES({col})" for col in df.columns])};
            """

            cursor.executemany(insert_query, df.fillna("NULL").values.tolist())
            conn.commit()

            print(f"✅ {table_name}: {cursor.rowcount} rows inserted/updated.")
            open(file_path, "w").close()
            print(f"🧹 Cleared {table_name}.csv after successful sync.")

        cursor.close()
        conn.close()
        print("🎯 Push complete.\n")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("❌ Access denied — check credentials.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("❌ Database does not exist.")
        else:
            print(f"❌ MySQL Error: {err}")


# === CLEANUP ===
def cleanup(sig=None, frame=None):
    print("\n🛑 Exiting... Closing CSVs and DB connections.")
    for f in csv_files.values():
        f.close()
    os._exit(0)


# === BACKGROUND THREAD ===
def start_ingestion_background(sleep_interval=5):
    def _loop():
        print("🔍 Background ingestion started...")
        file_positions = {}
        csv_totals = {t: 0 for t in CSV_HEADERS}

        while True:
            parsed_counts = {}
            for filename in os.listdir(LOG_DIR):
                if not filename.endswith(".log"):
                    continue
                path = os.path.join(LOG_DIR, filename)
                last_pos = file_positions.get(filename, 0)
                new_lines = 0

                with open(path, "r") as f:
                    f.seek(last_pos)
                    for line in f:
                        parsed = parse_line(filename, line)
                        if parsed:
                            table, values = parsed
                            write_csv_row(table, values)
                            csv_totals[table] += 1
                            new_lines += 1
                    file_positions[filename] = f.tell()
                parsed_counts[filename] = new_lines

            print("\n📊 Cycle Summary:")
            for log_file, count in parsed_counts.items():
                table = next((t for k, t in {
                    "api_event": "api_event_log",
                    "berth": "berth_application_log",
                    "container": "container_service_log",
                    "vessel_advice": "vessel_advice_log",
                    "vessel_registry": "vessel_registry_log",
                    "edi_advice": "edi_advice_log"
                }.items() if k in log_file), None)
                total = csv_totals.get(table, 0)
                print(f"✅ {log_file}: {count} new line(s) parsed (total {total})")
            print("-" * 80)

            push_csv_to_mysql()
            time.sleep(sleep_interval)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


# === MAIN (standalone mode) ===
if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    start_ingestion_background()
    while True:
        time.sleep(1)
