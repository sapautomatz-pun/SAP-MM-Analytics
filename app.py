# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI App (v16.0)
# ==========================================================
# Features:
#  ✅ AI Insights (GPT-4o)
#  ✅ Branded PDF & HTML Email
#  ✅ Airtable storage of sent reports
#  ✅ “My Reports” Portal for customers
#  ✅ Test PDF Preview mode
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI
from fpdf import FPDF

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
BACKEND_URL = "https://sapautomatz-backend.onrender.com"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY")
MAILER_FROM_EMAIL = "sapautomatz@gmail.com"
MAILER_FROM_NAME = "SAP Automatz"
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

# Airtable setup
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Reports")

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------
# STREAMLIT SETTINGS
# ----------------------------------------------------------
st.set_page_config(page_title="SAP Automatz - Procurement Analytics AI", page_icon="📊", layout="wide")
st.markdown("""<style>.main {background-color: #F6F9FC;} div.block-container {padding-top: 2rem;}</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------
# HEADER + NAVIGATION
# ----------------------------------------------------------
st.sidebar.image(LOGO_URL, width=150)
page = st.sidebar.radio("📘 Navigate", ["🔍 Analyze & Generate", "📬 My Reports Portal"])
st.sidebar.caption("Automate. Analyze. Accelerate 🚀")

# ----------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------
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

def calculate_kpis(df):
    kpis = {}
    kpis["records"] = len(df)
    kpis["total_spend"] = float(df["VALUE"].sum()) if "VALUE" in df.columns else 0
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
# AI ANALYSIS
# ----------------------------------------------------------
def build_prompt(kpis):
    return f"""
Analyze KPIs:
Total Records: {kpis['records']}
Total Spend: ₹{kpis['total_spend']:,}
Average Cycle Time: {kpis['avg_cycle']} days
Delayed Shipments: {kpis['delays']}
Top Vendor: {kpis['top_vendor']}
Provide:
1️⃣ Executive Summary
2️⃣ Root Causes
3️⃣ Recommendations
4️⃣ KPIs for next month
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
# PDF GENERATOR
# ----------------------------------------------------------
class BrandedPDF(FPDF):
    def header(self):
        self.set_fill_color(33, 86, 145)
        self.rect(0, 0, 210, 20, 'F')
        self.image(LOGO_URL, 10, 2, 25)
        self.set_text_color(255,255,255)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "SAP Automatz - Procurement Analytics Report", align="C", ln=True)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(130,130,130)
        self.cell(0, 10, "© 2025 SAP Automatz – Powered by Gen AI", align="C")

def sanitize_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^\x20-\x7E]+', ' ', str(text))
    return text.strip()[:200]

def generate_pdf(ai_text, kpis):
    pdf = BrandedPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key KPIs", ln=True)
    pdf.set_font("Helvetica", size=11)
    for k, v in kpis.items():
        pdf.multi_cell(0, 8, f"{sanitize_text(k).title()}: {sanitize_text(v)}")
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "AI Insight Summary", ln=True)
    pdf.set_font("Helvetica", size=11)
    for line in sanitize_text(ai_text).split(". "):
        pdf.multi_cell(0, 6, line.strip())
    pdf_bytes = bytes(pdf.output(dest='S').encode('latin-1', errors='ignore'))
    return io.BytesIO(pdf_bytes)

# ----------------------------------------------------------
# AIRTABLE STORAGE
# ----------------------------------------------------------
def store_report_in_airtable(email, report_name, url):
    endpoint = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    data = {"records": [{"fields": {"Email": email, "Report Name": report_name, "Report URL": url, "Created On": str(datetime.date.today())}}]}
    r = requests.post(endpoint, headers=headers, json=data)
    return r.status_code in [200, 201]

def fetch_reports_from_airtable(email):
    endpoint = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}?filterByFormula=FIND('{email}',{{Email}})"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    r = requests.get(endpoint, headers=headers)
    if r.status_code == 200:
        data = r.json()
        return [{"Report Name": rec["fields"].get("Report Name", ""), "Report URL": rec["fields"].get("Report URL", ""), "Created On": rec["fields"].get("Created On", "")} for rec in data.get("records", [])]
    return []

# ----------------------------------------------------------
# EMAIL (HTML)
# ----------------------------------------------------------
def build_html_email(customer_name, report_url):
    return f"""
    <html>
    <body style="font-family: Arial; background:#f6f9fc; padding:20px;">
      <div style="max-width:600px;margin:auto;background:#fff;border-radius:10px;padding:20px;">
        <div align="center">
          <img src="{LOGO_URL}" width="120"><h2 style="color:#215691;">SAP Automatz</h2>
          <p>Automate. Analyze. Accelerate.</p>
        </div>
        <p>Dear {customer_name},</p>
        <p>Your <b>Procurement Analytics Report</b> has been generated.</p>
        <p align="center">
          <a href="{report_url}" style="background:#215691;color:#fff;padding:12px 25px;border-radius:8px;text-decoration:none;">📄 Download Report</a>
        </p>
        <p>Thank you for using SAP Automatz!</p>
      </div>
    </body>
    </html>
    """

# ----------------------------------------------------------
# PAGE 1: ANALYZE & GENERATE
# ----------------------------------------------------------
if page == "🔍 Analyze & Generate":
    uploaded_file = st.file_uploader("📁 Upload your SAP Procurement Data", type=["csv", "xlsx"])
    recipient = st.text_input("✉️ Enter Customer Email")

    if uploaded_file and recipient:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
        df = normalize_columns(coerce_types(df))
        kpis, df = calculate_kpis(df)
        ai_text = generate_ai_summary(build_prompt(kpis))
        pdf_bytes = generate_pdf(ai_text, kpis)
        report_name = f"SAP_Report_{datetime.date.today()}.pdf"
        download_url = f"https://sapautomatz.streamlit.app/reports/{report_name}"

        # Save record
        store_report_in_airtable(recipient, report_name, download_url)
        st.success("✅ Report generated and stored in Airtable!")

        # Email to customer
        payload = {
            "from": {"email": MAILER_FROM_EMAIL, "name": MAILER_FROM_NAME},
            "to": [{"email": recipient}],
            "subject": "Your SAP Automatz Procurement Report",
            "html": build_html_email(recipient, download_url)
        }
        files = {'attachments': (report_name, pdf_bytes, 'application/pdf')}
        headers = {"Authorization": f"Bearer {MAILERSEND_API_KEY}"}
        requests.post("https://api.mailersend.com/v1/email", headers=headers, data={"message": json.dumps(payload)}, files=files)

        st.download_button("📄 Download Report", pdf_bytes, report_name, "application/pdf")

# ----------------------------------------------------------
# PAGE 2: CUSTOMER REPORT PORTAL
# ----------------------------------------------------------
if page == "📬 My Reports Portal":
    st.markdown("### 📦 Access Your Reports")
    email = st.text_input("Enter your registered email to view reports")
    if st.button("🔍 Fetch My Reports"):
        if email:
            reports = fetch_reports_from_airtable(email)
            if reports:
                st.success(f"Found {len(reports)} reports for {email}")
                for r in reports:
                    st.markdown(f"📄 **{r['Report Name']}** — _{r['Created On']}_")
                    st.markdown(f"[🔗 Download Report]({r['Report URL']})", unsafe_allow_html=True)
            else:
                st.info("No reports found for this email yet.")
        else:
            st.warning("Please enter a valid email address.")
