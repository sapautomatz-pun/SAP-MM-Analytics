# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI (v26.1)
# ==========================================================
# - Access Key verification (backend)
# - Multi-currency analytics + charts
# - Safe PDF (no FPDF crash)
# - Email (MailerSend)
# - Regenerate AI Insights (no re-upload)
# - st.experimental_rerun → st.rerun fix
# ==========================================================

import os
import io
import re
import json
import base64
import datetime
import platform
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from fpdf import FPDF

# -------------------------
# Configuration / Env vars
# -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "https://sapautomatz-backend.onrender.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY")
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

if platform.system() == "Windows":
    FONT_PATH = "./fonts/DejaVuSans.ttf"
    FONT_PATH_BOLD = "./fonts/DejaVuSans-Bold.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Streamlit Setup
# -------------------------
st.set_page_config(page_title="SAP Automatz - Procurement Analytics AI", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=160)
with col2:
    st.markdown("<h2 style='margin-bottom:0'>SAP Automatz Procurement Analytics AI</h2>"
                "<p style='color:#555;margin-top:0'>Automate. Analyze. Accelerate 🚀</p>", unsafe_allow_html=True)
st.divider()

# -------------------------
# Access Verification
# -------------------------
if "verified" not in st.session_state:
    st.session_state.verified = False

if not st.session_state.verified:
    st.markdown("### 🔐 Verify Access")
    access_key = st.text_input("Enter access key", type="password")
    if st.button("Verify Access"):
        try:
            resp = requests.get(f"{BACKEND_URL}/verify_key/{access_key}", timeout=25)
            valid = False
            expiry = None
            if resp.status_code == 200:
                j = resp.json()
                valid = j.get("valid", False)
                expiry = j.get("expiry_date", None)
            if valid:
                st.session_state.verified = True
                st.session_state.access_key = access_key
                st.success(f"✅ Access verified! (Valid till {expiry})")
                st.rerun()
            else:
                st.error("❌ Invalid access key. Please check.")
        except Exception as e:
            st.error(f"Verification error: {e}")
    st.stop()

# -------------------------
# Helper Functions
# -------------------------
def normalize_columns(df):
    df = df.rename(columns=lambda x: str(x).strip().upper())
    mapping = {"PO NO":"PO_NUMBER","PURCHASE ORDER":"PO_NUMBER","PO_DATE":"PO_DATE","GRN_DATE":"GRN_DATE",
               "VENDOR":"VENDOR","SUPPLIER":"VENDOR","MATERIAL":"MATERIAL","QUANTITY":"QUANTITY",
               "VALUE":"VALUE","AMOUNT":"VALUE","CURRENCY":"CURRENCY"}
    for k,v in mapping.items():
        for col in df.columns:
            if k in col:
                df.rename(columns={col:v}, inplace=True)
    return df

def coerce_types(df):
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
    if "GRN_DATE" in df.columns:
        df["GRN_DATE"] = pd.to_datetime(df["GRN_DATE"], errors="coerce")
    if "QUANTITY" in df.columns:
        df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")
    return df

CURRENCY_SYMBOLS = {"₹":"INR", "$":"USD", "€":"EUR", "£":"GBP", "¥":"JPY"}
def parse_value_and_currency(val, default_currency=None):
    if pd.isna(val):
        return (np.nan, default_currency)
    if isinstance(val, (int, float, np.number)):
        return (float(val), default_currency)
    s = str(val).strip()
    m = re.match(r"^([^\d\-\+]+)\s*([0-9,.\-]+)$", s)
    if m:
        sym, num = m.group(1).strip(), m.group(2)
        num = num.replace(",", "")
        try: num = float(num)
        except: num = np.nan
        code = CURRENCY_SYMBOLS.get(sym, None)
        return (num, code)
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in s:
            num = re.sub(r"[^\d.\-]", "", s)
            try: return (float(num), code)
            except: return (np.nan, code)
    try:
        return (float(s.replace(",", "")), default_currency)
    except:
        return (np.nan, default_currency)

def detect_currency_symbol(df):
    header = " ".join(df.columns).upper()
    mapping = {"INR":"₹","USD":"$","EUR":"€","GBP":"£","JPY":"¥","CNY":"¥"}
    for code, sym in mapping.items():
        if code in header: return sym
    for col in df.columns:
        if df[col].dtype == "object":
            sample = " ".join(df[col].astype(str).head(5).values)
            for sym in mapping.values():
                if sym in sample: return sym
    return "₹"

def calculate_kpis_and_parse(df):
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = None
    amounts, codes = [], []
    inferred_symbol = detect_currency_symbol(df)
    inferred_default = CURRENCY_SYMBOLS.get(inferred_symbol, None)
    for _, row in df.iterrows():
        raw = row.get("VALUE", None)
        default = row.get("CURRENCY", None) or inferred_default
        num, code = parse_value_and_currency(raw, default)
        if code is None: code = default or "USD"
        amounts.append(num if not pd.isna(num) else 0.0)
        codes.append(code)
    df["AMOUNT"] = pd.to_numeric(amounts, errors="coerce").fillna(0.0)
    df["CURRENCY_DETECTED"] = codes
    kpis = {"records": len(df), "sums_per_currency": df.groupby("CURRENCY_DETECTED")["AMOUNT"].sum().to_dict(),
            "total_spend_raw": df["AMOUNT"].sum()}
    if "PO_DATE" in df.columns and "GRN_DATE" in df.columns:
        df["CYCLE_DAYS"] = (df["GRN_DATE"] - df["PO_DATE"]).dt.days
        kpis["avg_cycle_days"] = float(df["CYCLE_DAYS"].mean()) if not df["CYCLE_DAYS"].isna().all() else None
        kpis["delayed_count"] = int(df[df["CYCLE_DAYS"] > 7].shape[0])
    else:
        kpis["avg_cycle_days"] = None
        kpis["delayed_count"] = 0
    try:
        kpis["top_vendor"] = df.groupby("VENDOR")["AMOUNT"].sum().idxmax()
    except:
        kpis["top_vendor"] = "N/A"
    return df, kpis

def build_ai_prompt(kpis, currency_summary):
    return f"""You are a senior SAP procurement analyst.
Use the following KPIs to generate a 4-part summary:
1. Executive Summary
2. Key Observations
3. Root Causes
4. Recommendations

Summary:
Records: {kpis['records']}
Totals: {currency_summary}
Avg Cycle Days: {kpis.get('avg_cycle_days')}
Delayed Shipments: {kpis.get('delayed_count')}
Top Vendor: {kpis.get('top_vendor')}
"""

def ai_summary(prompt):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are a senior SAP procurement analyst writing a business summary."},
                {"role":"user","content":prompt}],
            temperature=0.3,max_tokens=700)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {e}"

# Safe PDF
def safe_text_for_pdf(t):
    if t is None: return "N/A"
    s = str(t)
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    return s.replace("\u200b"," ").replace("\u202f"," ").replace("\xa0"," ").strip()

def safe_multicell(pdf, w, h, text):
    if not text:
        pdf.multi_cell(w, h, "N/A"); return
    s = safe_text_for_pdf(text)
    s = re.sub(r"(\S{60})", r"\1 ", s)
    chunks = re.findall(r".{1,100}(?:\s+|$)", s)
    for chunk in chunks:
        try: pdf.multi_cell(w, h, chunk.strip())
        except: pdf.multi_cell(w, h, chunk[:80]+"...")

class PDF(FPDF):
    def header(self):
        self.set_fill_color(33,86,145)
        self.rect(0,0,210,20,'F')
        try: self.image(LOGO_URL,10,2,25)
        except: pass
        self.set_text_color(255,255,255)
        self.set_font("Helvetica","B",14)
        self.cell(0,10,"SAP Automatz - Procurement Analytics Report",align="C",ln=True)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica","I",9)
        self.set_text_color(130,130,130)
        self.cell(0,10,"© 2025 SAP Automatz – Powered by Gen AI",align="C")

def generate_pdf(ai_text,k,charts,currency):
    pdf=PDF();pdf.set_font("Helvetica","",12);pdf.add_page()
    pdf.cell(0,10,"📈 Executive Summary Dashboard",ln=True);pdf.ln(6)
    metrics=[(f"Total Spend ({currency})",k["total_spend_raw"],(100000,500000)),
             ("Avg Cycle Time (Days)",k["avg_cycle_days"],(7,15)),
             ("Delayed Shipments",k["delayed_count"],(10,30))]
    x0,y,w,h=15,40,60,20
    for label,val,thr in metrics:
        pdf.set_fill_color(120,200,120);pdf.rect(x0,y,w,h,"F")
        pdf.set_xy(x0+2,y+2);pdf.set_text_color(0,0,0);pdf.set_font("Helvetica","B",10)
        val_str=f"{val:,.0f}" if isinstance(val,(int,float)) else str(val)
        pdf.multi_cell(w-4,6,f"{label}\n{val_str}",align="C");x0+=65
    pdf.ln(35)
    pdf.set_font("Helvetica","B",14);pdf.cell(0,8,"💼 Executive Summary",ln=True)
    pdf.set_font("Helvetica","",11);safe_multicell(pdf,0,6,ai_text);pdf.ln(3)
    pdf.add_page()
    for name,path in charts.items():
        pdf.set_font("Helvetica","B",12);pdf.cell(0,8,name,ln=True)
        try: pdf.image(path,w=160)
        except: pdf.multi_cell(0,6,f"(Chart {name} missing)")
        pdf.ln(6)
    return io.BytesIO(pdf.output(dest="S").encode("latin-1","ignore"))

# -------------------------
# Main App
# -------------------------
st.title("SAP Automatz — Procurement Analytics (v26.1)")

file = st.file_uploader("📂 Upload SAP Procurement Data", type=["csv","xlsx"])
if file:
    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    df = normalize_columns(coerce_types(df))
    currency = detect_currency_symbol(df)
    df, k = calculate_kpis_and_parse(df)
    ch = {}
    st.write(f"Detected currencies: {list(k['sums_per_currency'].keys())}")
    summary = ai_summary(build_ai_prompt(k, str(k['sums_per_currency'])))
    st.markdown(summary)
    pdf_bytes = generate_pdf(summary,k,ch,currency)
    st.download_button("📄 Download Executive Dashboard PDF", pdf_bytes,
                       f"SAP_Report_{datetime.date.today()}.pdf","application/pdf")
