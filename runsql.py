# runsql.py
import mysql.connector
import pandas as pd
import streamlit as st

# -----------------------------------
# 1️⃣ Database Connection Function
# -----------------------------------
def get_connection():
    """Creates and returns a MySQL connection."""
    return mysql.connector.connect(
        host="psahackathon.ckn0s0ok2sbi.us-east-1.rds.amazonaws.com",
        user="admin",
        password="t0500224E",  # your actual RDS password
        database="appdb",
        port=3306,
        ssl_disabled=False
    )

# -----------------------------------
# 2️⃣ Run Query Function
# -----------------------------------
def run_query(sql: str) -> pd.DataFrame:
    """Executes SQL and returns results as DataFrame."""
    try:
        conn = get_connection()
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"⚠️ Error running query: {e}")
        return pd.DataFrame()

# -----------------------------------
# 3️⃣ Push CSVs to MySQL (optional)
# -----------------------------------
def push_csvs_to_mysql(directory: str = "."):
    """Pushes all .csv files in a directory into MySQL tables."""
    import os

    results = []
    try:
        conn = get_connection()
        cursor = conn.cursor()

        for file in os.listdir(directory):
            if not file.endswith(".csv"):
                continue

            table_name = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(directory, file))

            if df.empty:
                continue

            cols = ", ".join(df.columns)
            placeholders = ", ".join(["%s"] * len(df.columns))
            update_stmt = ", ".join([f"{col}=VALUES({col})" for col in df.columns])

            sql = f"""
                INSERT INTO {table_name} ({cols})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_stmt};
            """

            cursor.executemany(sql, df.fillna("NULL").values.tolist())
            conn.commit()
            results.append((table_name, len(df)))

        cursor.close()
        conn.close()
        return results

    except Exception as e:
        st.error(f"❌ MySQL push error: {e}")
        return []
