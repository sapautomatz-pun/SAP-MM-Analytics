# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI App (Streamlit v13.0)
# ==========================================================
# New Features:
#  ✅ Branded Header/Footer in PDF
#  ✅ Vendor / Material / Monthly Trend Charts
#  ✅ AI Insights (GPT-4o)
#  ✅ One-Click Branded PDF Export
#  ✅ Email Report to Customer (MailerSend API)
# ==========================================================

import os
import io
import re
import json
import requests
import pandas as pd
import numpy as np
import datetime
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI
from fpdf import FPDF

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
BACKEND_URL = "https://sapautomatz-backend.onrender.com"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY") or "mlsn-your-key-here"
MAILER_FROM_EMAIL = "sapautomatz@gmail.com"
MAILER_FROM_NAME = "SAP Automatz"
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
# LOGO + TITLE
# ----------------------------------------------------------
st.image("https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png", width=180)
st.title("SAP Automatz - Procurement Analytics AI")
st.caption("Automate. Analyze. Accelerate 🚀")

# ----------------------------------------------------------
# ACCESS VALIDATION
# ----------------------------------------------------------
st.markdown("#### 🔐 Step 1: Verify your Access Key")
access_key = st.text_input("Enter your Access Key (from your purchase email)", type="password")

if access_key:
    try:
        res = requests.get(f"{BACKEND_URL}/verify_access", params={"key": access_key}, timeout=10)
        data = res.json()
        if res.status_code == 200 and data.get("status") == "ok":
            st.success(f"✅ Access granted to {data.get('name')} ({data.get('plan').capitalize()} plan, valid until {data.get('expiry')})")
            valid_access = True
        else:
            st.error("❌ Invalid or expired Access Key.")
            valid_access = False
    except Exception as e:
        st.error(f"Error verifying access: {e}")
        valid_access = False
else:
    valid_access = False

if not valid_access:
    st.stop()

# ----------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------
st.markdown("### 📁 Step 2: Upload your SAP Procurement Data")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

# ----------------------------------------------------------
# DATA UTILITIES
# ----------------------------------------------------------
def load_uploaded_file(uploaded_file):
    return pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

def normalize_columns(df):
    df = df.rename(columns=lambda x: str(x).strip().upper())
    mapping = {
        "PO NO": "PO_NUMBER", "PURCHASE ORDER": "PO_NUMBER",
        "PO_DATE": "PO_DATE", "GRN_DATE": "GRN_DATE",
        "VENDOR": "VENDOR", "SUPPLIER": "VENDOR",
        "MATERIAL": "MATERIAL", "QUANTITY": "QUANTITY",
        "VALUE": "VALUE", "AMOUNT": "VALUE"
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
# CHARTS
# ----------------------------------------------------------
def generate_charts(df):
    chart_paths = {}

    # Vendors
    if "VENDOR" in df.columns and "VALUE" in df.columns:
        top_vendors = df.groupby("VENDOR")["VALUE"].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(8,4))
        top_vendors.plot(kind='bar', color='steelblue', title="Top 10 Vendors by Spend (₹)")
        plt.ylabel("Spend (₹)")
        plt.tight_layout()
        vendor_chart = "/tmp/vendor_chart.png"
        plt.savefig(vendor_chart)
        chart_paths["Top Vendors"] = vendor_chart
        st.image(vendor_chart, caption="Top Vendors by Spend")

    # Materials
    if "MATERIAL" in df.columns and "VALUE" in df.columns:
        material_spend = df.groupby("MATERIAL")["VALUE"].sum().sort_values(ascending=False).head(8)
        plt.figure(figsize=(5,5))
        plt.pie(material_spend, labels=material_spend.index, autopct="%1.1f%%")
        plt.title("Material Spend Distribution")
        plt.tight_layout()
        material_chart = "/tmp/material_chart.png"
        plt.savefig(material_chart)
        chart_paths["Material Spend"] = material_chart
        st.image(material_chart, caption="Material Spend Distribution")

    # Monthly Trend
    if "PO_DATE" in df.columns and "VALUE" in df.columns:
        df["MONTH"] = df["PO_DATE"].dt.to_period("M").astype(str)
        monthly_spend = df.groupby("MONTH")["VALUE"].sum().sort_index()
        plt.figure(figsize=(8,4))
        monthly_spend.plot(marker='o', color='darkorange', title="Monthly Spend Trend")
        plt.ylabel("Spend (₹)")
        plt.tight_layout()
        trend_chart = "/tmp/trend_chart.png"
        plt.savefig(trend_chart)
        chart_paths["Monthly Trend"] = trend_chart
        st.image(trend_chart, caption="Monthly Spend Trend")

    return chart_paths

# ----------------------------------------------------------
# AI ANALYSIS
# ----------------------------------------------------------
def build_prompt(kpis, df):
    return f"""
You are a procurement analytics assistant.
Analyze these KPIs and dataset:
Total Records: {kpis['records']}
Total Spend: ₹{kpis['total_spend']:,}
Average Cycle Time: {kpis['avg_cycle']} days
Delayed Shipments: {kpis['delays']}
Top Vendor: {kpis['top_vendor']}
Provide insights in 4 short sections:
1️⃣ Executive Summary
2️⃣ Root Causes
3️⃣ Actionable Recommendations
4️⃣ Suggested KPIs for next month.
Keep under 350 words.
"""

def generate_ai_summary(prompt):
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
# PDF GENERATION (Branded)
# ----------------------------------------------------------
class BrandedPDF(FPDF):
    def header(self):
        self.set_fill_color(33, 86, 145)
        self.rect(0, 0, 210, 20, 'F')
        self.image("https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png", 10, 2, 25)
        self.set_text_color(255,255,255)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "SAP Automatz - Procurement Analytics Report", align="C", ln=True)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(130,130,130)
        self.cell(0, 10, "© 2025 SAP Automatz – Powered by Gen AI", align="C")

def generate_pdf(ai_text, kpis, df, chart_paths):
    pdf = BrandedPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key KPIs", ln=True)
    pdf.set_font("Helvetica", size=11)
    for k, v in kpis.items():
        pdf.multi_cell(0, 8, f"{k.title()}: {v}")
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "AI Insight Summary", ln=True)
    pdf.set_font("Helvetica", size=11)
    for line in ai_text.split("\n"):
        pdf.multi_cell(0, 6, line)
    pdf.ln(6)
    for name, path in chart_paths.items():
        pdf.cell(0, 8, f"{name} Chart", ln=True)
        pdf.image(path, w=160)
        pdf.ln(6)
    pdf_bytes = bytes(pdf.output(dest='S').encode('latin-1', errors='ignore'))
    return io.BytesIO(pdf_bytes)

# ----------------------------------------------------------
# EMAIL SENDING
# ----------------------------------------------------------
def send_report_via_email(email, pdf_bytes):
    try:
        files = {'attachments': ('SAP_Procurement_Report.pdf', pdf_bytes, 'application/pdf')}
        payload = {
            "from": {"email": MAILER_FROM_EMAIL, "name": MAILER_FROM_NAME},
            "to": [{"email": email}],
            "subject": "Your SAP Automatz Procurement Analytics Report",
            "text": "Dear Customer,\n\nPlease find attached your SAP Procurement Analytics report generated by SAP Automatz.\n\nRegards,\nSAP Automatz Team",
        }
        headers = {"Authorization": f"Bearer {MAILERSEND_API_KEY}", "Content-Type": "application/json"}
        response = requests.post("https://api.mailersend.com/v1/email", headers=headers, json=payload, files=files)
        if response.status_code in [200, 202]:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# ----------------------------------------------------------
# MAIN APP FLOW
# ----------------------------------------------------------
if uploaded_file is not None:
    df = load_uploaded_file(uploaded_file)
    df = normalize_columns(df)
    df = coerce_types(df)

    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head(10))

    with st.spinner("Analyzing data..."):
        kpis, df = calculate_kpis(df)
        charts = generate_charts(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", kpis["records"])
    col2.metric("Total Spend (₹)", f"{kpis['total_spend']:,.0f}" if kpis["total_spend"] else "N/A")
    col3.metric("Avg Cycle Time (days)", kpis["avg_cycle"])
    col4.metric("Delayed Shipments", kpis["delays"])

    st.markdown("### 🧠 AI-Generated Insights")
    with st.spinner("Generating AI summary..."):
        ai_text = generate_ai_summary(build_prompt(kpis, df))
        st.success("AI analysis complete ✅")
        st.markdown(ai_text)

    if st.button("📄 Generate PDF Report"):
        pdf_bytes = generate_pdf(ai_text, kpis, df, charts)
        st.download_button("Download Report", pdf_bytes, "SAP_Procurement_Report.pdf", "application/pdf")

        st.markdown("---")
        st.markdown("### ✉️ Email this report to your customer")
        recipient_email = st.text_input("Enter recipient email address")
        if st.button("Send Report via Email"):
            if recipient_email:
                if send_report_via_email(recipient_email, pdf_bytes.getvalue()):
                    st.success(f"✅ Report sent successfully to {recipient_email}")
                else:
                    st.error("❌ Failed to send report. Please verify your MailerSend API key.")
            else:
                st.warning("Please enter a valid recipient email address.")

else:
    st.info("Please upload a SAP PO or GRN file to continue.")
