# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI App (Streamlit v8.0)
# ==========================================================
# Features:
#  - Access validation (via backend /verify_access)
#  - Upload SAP PO / GRN file (CSV/XLSX)
#  - KPI extraction (pandas)
#  - AI insight generation (OpenAI GPT)
#  - PDF report export
#  - Compatible with openai>=1.0 and Streamlit Cloud
# ==========================================================

import os
import io
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import streamlit as st
from openai import OpenAI

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
BACKEND_URL = "https://sapautomatz-backend.onrender.com"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ----------------------------------------------------------
st.set_page_config(page_title="SAP Automatz - Procurement Analytics AI",
                   page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #F6F9FC;}
    div.block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# VERIFY ACCESS
# ----------------------------------------------------------
st.image("https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png", width=180)
st.title("SAP Automatz - Procurement Analytics AI")

st.markdown("#### 🔐 Step 1: Verify your Access Key")

access_key = st.text_input("Enter your Access Key (from your purchase email)", type="password")

if access_key:
    try:
        res = requests.get(f"{BACKEND_URL}/verify_access", params={"key": access_key}, timeout=10)
        data = res.json()
        if res.status_code == 200 and data.get("status") == "ok":
            st.success(f"✅ Access granted to {data.get('name')} ({data.get('plan').capitalize()} plan, valid until {data.get('expiry')})")
            valid_access = True
        elif res.status_code == 403:
            st.error("❌ Access expired. Please renew your subscription.")
            valid_access = False
        else:
            st.error("❌ Invalid Access Key.")
            valid_access = False
    except Exception as e:
        st.error(f"Error verifying access: {e}")
        valid_access = False
else:
    valid_access = False

if not valid_access:
    st.stop()

# ----------------------------------------------------------
# OPENAI CONNECTION TEST
# ----------------------------------------------------------
st.markdown("### 🤖 Test OpenAI Connection")
if st.button("🔌 Test Connection to OpenAI"):
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY is not set. Please check Streamlit Secrets.")
    else:
        try:
            models = client.models.list()
            model_names = [m.id for m in models.data if "gpt" in m.id]
            st.success(f"✅ Connected successfully! Models available: {', '.join(model_names[:3])}")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

# ----------------------------------------------------------
# STEP 2: FILE UPLOAD
# ----------------------------------------------------------
st.markdown("### 📁 Step 2: Upload your SAP Procurement Data")
st.write("Upload your SAP Purchase Order or GRN file (.csv or .xlsx).")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

# ----------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------
def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

def normalize_columns(df):
    df = df.rename(columns=lambda x: str(x).strip().upper())
    mapping = {
        "PO NO": "PO_NUMBER",
        "PURCHASE ORDER": "PO_NUMBER",
        "PO_DATE": "PO_DATE",
        "GRN_DATE": "GRN_DATE",
        "VENDOR": "VENDOR",
        "SUPPLIER": "VENDOR",
        "MATERIAL": "MATERIAL",
        "QUANTITY": "QUANTITY",
        "VALUE": "VALUE",
        "AMOUNT": "VALUE"
    }
    for key, val in mapping.items():
        for col in df.columns:
            if key in col:
                df.rename(columns={col: val}, inplace=True)
    return df

def coerce_types(df):
    for c in ["PO_DATE", "GRN_DATE"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["QUANTITY", "VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------------------------------------
# KPI CALCULATION
# ----------------------------------------------------------
def calculate_kpis(df):
    kpis = {}
    kpis["records"] = len(df)
    kpis["total_spend"] = float(df["VALUE"].sum()) if "VALUE" in df.columns else None
    if "PO_DATE" in df.columns and "GRN_DATE" in df.columns:
        df["CYCLE_DAYS"] = (df["GRN_DATE"] - df["PO_DATE"]).dt.days
        kpis["avg_cycle"] = round(df["CYCLE_DAYS"].mean(), 1)
        kpis["delays"] = len(df[df["CYCLE_DAYS"] > 7])
    else:
        kpis["avg_cycle"], kpis["delays"] = None, 0
    if "VENDOR" in df.columns and "VALUE" in df.columns:
        kpis["top_vendor"] = df.groupby("VENDOR")["VALUE"].sum().idxmax()
    else:
        kpis["top_vendor"] = "N/A"
    return kpis, df

# ----------------------------------------------------------
# AI PROMPT GENERATION
# ----------------------------------------------------------
def build_prompt(kpis, df):
    lines = [
        "You are a procurement analytics assistant.",
        "Analyze the following KPIs and dataset summary, then provide a business-level report:",
        "",
        f"Total Records: {kpis.get('records')}",
        f"Total Spend (₹): {kpis.get('total_spend', 'N/A')}",
        f"Average Cycle Time: {kpis.get('avg_cycle', 'N/A')} days",
        f"Delayed Shipments (>7 days): {kpis.get('delays', 0)}",
        f"Top Vendor by Spend: {kpis.get('top_vendor', 'N/A')}",
        "",
        "Sample Data (first 10 rows):",
        df.head(10).to_csv(index=False),
        "",
        "Now write:",
        "1️⃣ A 5-bullet Executive Summary.",
        "2️⃣ 3 key Root-Cause Hypotheses.",
        "3️⃣ 3 Actionable Recommendations.",
        "4️⃣ 3 Suggested KPIs to track next month.",
        "Keep under 350 words. Use simple business language."
    ]
    return "\n".join(lines)

# ----------------------------------------------------------
# NEW OPENAI API CALL (v1.x)
# ----------------------------------------------------------
def generate_ai_summary(prompt):
    if not OPENAI_API_KEY:
        return "⚠️ No OpenAI API key configured."
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a senior procurement analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating AI summary: {e}"

# ----------------------------------------------------------
# FIXED PDF GENERATION
# ----------------------------------------------------------
def generate_pdf(ai_text, kpis, df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SAP Automatz - Procurement Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Key KPIs", ln=True)
    pdf.set_font("Arial", size=11)
    for k, v in kpis.items():
        pdf.cell(0, 8, f"{k.title()}: {v}", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "AI Insight Summary", ln=True)
    pdf.set_font("Arial", size=11)
    for line in ai_text.split("\n"):
        pdf.multi_cell(0, 6, line)
    
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return io.BytesIO(pdf_bytes)

# ----------------------------------------------------------
# MAIN APP FLOW
# ----------------------------------------------------------
if uploaded_file is not None:
    df = load_uploaded_file(uploaded_file)
    df = normalize_columns(df)
    df = coerce_types(df)

    if df.empty:
        st.error("Uploaded file has no valid data.")
        st.stop()

    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head(10))

    with st.spinner("Analyzing data..."):
        kpis, df = calculate_kpis(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", kpis["records"])
    col2.metric("Total Spend (₹)", f"{kpis['total_spend']:,.0f}" if kpis["total_spend"] else "N/A")
    col3.metric("Avg Cycle Time (days)", kpis["avg_cycle"] if kpis["avg_cycle"] else "N/A")
    col4.metric("Delayed Shipments", kpis["delays"])

    st.markdown("### 🧠 AI-Generated Insights")
    with st.spinner("Generating AI summary..."):
        prompt = build_prompt(kpis, df)
        ai_text = generate_ai_summary(prompt)
        st.success("AI analysis complete ✅")
        st.markdown(ai_text)

    if st.button("📄 Generate PDF Report"):
        pdf_bytes = generate_pdf(ai_text, kpis, df)
        st.download_button("Download Report", pdf_bytes, "SAP_Procurement_Report.pdf", "application/pdf")

else:
    st.info("Please upload a SAP PO or GRN file to continue.")
