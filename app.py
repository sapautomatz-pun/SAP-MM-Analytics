# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI (v26.0)
# Features:
#  - Access Key verification (backend)
#  - Multi-currency analytics + charts
#  - Safe PDF generation (Unicode, DejaVu)
#  - Send generated PDF by email (MailerSend) if configured
#  - "Regenerate AI Insights" without re-uploading file
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
MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY")  # optional, for email delivery
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

# Font paths
if platform.system() == "Windows":
    FONT_PATH = "./fonts/DejaVuSans.ttf"
    FONT_PATH_BOLD = "./fonts/DejaVuSans-Bold.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(page_title="SAP Automatz - Procurement Analytics AI", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

# Header / branding
col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=160)
with col2:
    st.markdown("<h2 style='margin-bottom:0'>SAP Automatz Procurement Analytics AI</h2>"
                "<p style='color:#555;margin-top:0'>Automate. Analyze. Accelerate 🚀</p>", unsafe_allow_html=True)
st.divider()

# -------------------------
# Access verification
# -------------------------
if "verified" not in st.session_state:
    st.session_state.verified = False

if not st.session_state.verified:
    st.markdown("### 🔐 Verify Access")
    access_key = st.text_input("Enter access key", type="password")
    if st.button("Verify Access"):
        try:
            # example endpoint: GET BACKEND_URL/verify_key/{key} returning JSON {"valid": true/false, ...}
            resp = requests.get(f"{BACKEND_URL}/verify_key/{access_key}", timeout=25)
            ok = False
            try:
                ok = resp.status_code == 200 and resp.json().get("valid", False)
            except:
                ok = False
            if ok:
                st.session_state.verified = True
                st.session_state.access_key = access_key
                st.success("✅ Access verified — you may continue.")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid access key. Please check.")
        except Exception as e:
            st.error(f"Verification error: {e}")
    st.stop()

# -------------------------
# Helper functions
# -------------------------
def normalize_columns(df):
    df = df.rename(columns=lambda x: str(x).strip().upper())
    mapping = {
        "PO NO": "PO_NUMBER", "PURCHASE ORDER": "PO_NUMBER",
        "PO_DATE": "PO_DATE", "GRN_DATE": "GRN_DATE",
        "VENDOR": "VENDOR", "SUPPLIER": "VENDOR",
        "MATERIAL": "MATERIAL", "QUANTITY": "QUANTITY",
        "VALUE": "VALUE", "AMOUNT": "VALUE", "CURRENCY": "CURRENCY"
    }
    for k, v in mapping.items():
        for col in list(df.columns):
            if k in col:
                df.rename(columns={col: v}, inplace=True)
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
    # symbol at start
    m = re.match(r"^([^\d\-\+]+)\s*([0-9,.\-]+)$", s)
    if m:
        sym, num = m.group(1).strip(), m.group(2)
        num = num.replace(",", "")
        try:
            num = float(num)
        except:
            num = np.nan
        code = CURRENCY_SYMBOLS.get(sym, None)
        return (num, code)
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in s:
            num = re.sub(r"[^\d.\-]", "", s)
            try:
                return (float(num), code)
            except:
                return (np.nan, code)
    try:
        return (float(s.replace(",", "")), default_currency)
    except:
        return (np.nan, default_currency)

def detect_currency_symbol(df):
    header = " ".join(df.columns).upper()
    mapping = {"INR":"₹","USD":"$","EUR":"€","GBP":"£","JPY":"¥","CNY":"¥"}
    for code, sym in mapping.items():
        if code in header:
            return sym
    for col in df.columns:
        if df[col].dtype == "object":
            sample = " ".join(df[col].astype(str).head(5).values)
            for sym in mapping.values():
                if sym in sample:
                    return sym
    return "₹"

def calculate_kpis_and_parse(df):
    # ensure currency column exists
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = None
    # parse amounts
    amounts = []
    codes = []
    inferred_symbol = detect_currency_symbol(df)
    inferred_default = CURRENCY_SYMBOLS.get(inferred_symbol, None)
    for _, row in df.iterrows():
        raw = row.get("VALUE", None)
        default = row.get("CURRENCY", None) or inferred_default
        num, code = parse_value_and_currency(raw, default)
        if code is None:
            code = default or "USD"
        amounts.append(num if not pd.isna(num) else 0.0)
        codes.append(code)
    df["AMOUNT"] = pd.to_numeric(amounts, errors="coerce").fillna(0.0)
    df["CURRENCY_DETECTED"] = codes
    kpis = {}
    kpis["records"] = len(df)
    kpis["sums_per_currency"] = df.groupby("CURRENCY_DETECTED")["AMOUNT"].sum().to_dict()
    kpis["total_spend_raw"] = df["AMOUNT"].sum()
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

def build_ai_prompt(kpis, currency_summary_str):
    return f"""
You are a senior procurement analyst. Use the summary below and write a concise executive summary and recommendations.

Summary:
- Total records: {kpis['records']}
- Totals by currency: {currency_summary_str}
- Average cycle days: {kpis.get('avg_cycle_days')}
- Delayed shipments (>7 days): {kpis.get('delayed_count')}
- Top vendor: {kpis.get('top_vendor')}

Provide four sections:
1) Executive Summary
2) Key Observations
3) Root Causes
4) Actionable Recommendations
Keep it concise and suitable for senior management.
"""

def get_ai_summary(prompt):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are an expert procurement analyst."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
            max_tokens=700
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {e}"

# Chart helpers
def save_vendor_chart(df, out_path="vendor_chart.png"):
    if "VENDOR" in df.columns:
        top = df.groupby("VENDOR")["AMOUNT"].sum().nlargest(10)
        plt.figure(figsize=(8,4)); top.plot(kind="bar", color="steelblue")
        plt.title("Top Vendors by Amount"); plt.tight_layout(); plt.savefig(out_path); plt.close()
        return out_path
    return None

def save_trend_chart(df, out_path="trend_chart.png"):
    if "PO_DATE" in df.columns:
        tmp = df.dropna(subset=["PO_DATE"])
        if not tmp.empty:
            tmp["MONTH"]=tmp["PO_DATE"].dt.to_period("M").astype(str)
            m = tmp.groupby("MONTH")["AMOUNT"].sum()
            plt.figure(figsize=(8,4)); m.plot(marker='o', color="darkorange"); plt.title("Monthly Spend Trend"); plt.tight_layout(); plt.savefig(out_path); plt.close()
            return out_path
    return None

def save_material_pie(df, out_path="material_chart.png"):
    if "MATERIAL" in df.columns:
        mat = df.groupby("MATERIAL")["AMOUNT"].sum().nlargest(6)
        if not mat.empty:
            plt.figure(figsize=(5,5)); plt.pie(mat, labels=mat.index, autopct="%1.1f%%"); plt.title("Material Spend"); plt.tight_layout(); plt.savefig(out_path); plt.close()
            return out_path
    return None

# -------------------------
# Safe PDF generation utilities
# -------------------------
def safe_text_for_pdf(t):
    if t is None:
        return "N/A"
    s = str(t)
    # remove control characters but keep Unicode (we will use DejaVu)
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    # replace zero-width or nbsp
    s = s.replace("\u200b"," ").replace("\u202f"," ").replace("\xa0"," ")
    return s.strip()

def safe_multicell(pdf, w, h, text):
    # Break text into safe chunks to avoid "Not enough horizontal space" errors
    if not text:
        pdf.multi_cell(w, h, "N/A")
        return
    s = safe_text_for_pdf(text)
    # ensure no crazy long tokens: insert spaces every 60 characters in sequences of non-space chars
    s = re.sub(r"(\S{60})", r"\1 ", s)
    # split by sentences/near 100 char chunks
    chunks = re.findall(r".{1,100}(?:\s+|$)", s)
    for chunk in chunks:
        c = chunk.strip()
        if c:
            try:
                pdf.multi_cell(w, h, c)
            except Exception:
                pdf.multi_cell(w, h, c[:80] + "...")

class DejaVuPDF(FPDF):
    def header(self):
        self.set_fill_color(33,86,145)
        self.rect(0,0,210,20,'F')
        try:
            self.image(LOGO_URL, 10, 2, 25)
        except:
            pass
        self.set_text_color(255,255,255)
        try:
            self.set_font("DejaVu","B",14)
        except:
            self.set_font("Helvetica","B",14)
        self.cell(0,10,"SAP Automatz - Procurement Analytics Report", align="C", ln=True)
        self.ln(2)
    def footer(self):
        self.set_y(-15)
        try:
            self.set_font("DejaVu","I",9)
        except:
            self.set_font("Helvetica","I",9)
        self.set_text_color(130,130,130)
        self.cell(0,10,"©2025 SAP Automatz – Powered by Gen AI", align="C")

def ensure_fonts(pdf):
    try:
        if os.path.exists(FONT_PATH):
            pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        if os.path.exists(FONT_PATH_BOLD):
            pdf.add_font("DejaVu", "B", FONT_PATH_BOLD, uni=True)
    except Exception:
        pass

def generate_pdf(ai_text, kpis, chart_paths, display_currency_symbol="₹"):
    pdf = DejaVuPDF()
    ensure_fonts(pdf)
    try:
        pdf.set_font("DejaVu","",12)
    except:
        pdf.set_font("Helvetica","",12)
    pdf.add_page()

    pdf.cell(0,10, "📈 Executive Summary Dashboard", ln=True)
    pdf.ln(6)

    # KPI boxes
    def kpi_color(v, good, warn):
        try:
            val = float(v)
        except:
            return (180,180,180)
        if val <= good: return (120,200,120)
        if val <= warn: return (255,210,80)
        return (255,100,100)

    metrics = [
        (f"Total Spend ({display_currency_symbol})", kpis.get("total_spend_raw", 0), (100000, 500000)),
        ("Avg Cycle Time (days)", kpis.get("avg_cycle_days", 0) or 0, (7, 15)),
        ("Delayed Shipments (>7d)", kpis.get("delayed_count", 0), (10, 30))
    ]
    x0, y, w, h = 15, 40, 60, 20
    for label, val, thr in metrics:
        r,g,b = kpi_color(val, thr[0], thr[1])
        pdf.set_fill_color(r,g,b)
        pdf.rect(x0, y, w, h, "F")
        pdf.set_xy(x0+2, y+2)
        pdf.set_text_color(0,0,0)
        try:
            pdf.set_font("DejaVu","B",11)
        except:
            pdf.set_font("Helvetica","B",11)
        val_str = f"{val:,.0f}" if isinstance(val, (int,float,np.number)) else str(val)
        pdf.multi_cell(w-4, 6, f"{label}\n{val_str}", align="C")
        x0 += (w + 5)
    pdf.ln(35)

    # AI Executive Summary
    try:
        pdf.set_font("DejaVu","B",14)
    except:
        pdf.set_font("Helvetica","B",14)
    pdf.cell(0,8, "💼 Executive Summary", ln=True)
    try:
        pdf.set_font("DejaVu","",11)
    except:
        pdf.set_font("Helvetica","",11)
    # write AI text with safe_multicell
    safe_multicell(pdf, 0, 6, ai_text)
    pdf.ln(6)

    # Multi-currency summary table (simple)
    try:
        pdf.set_font("DejaVu","B",12)
    except:
        pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8, "🌐 Totals by Currency (native)", ln=True)
    try:
        pdf.set_font("DejaVu","",11)
    except:
        pdf.set_font("Helvetica","",11)
    for cur, amt in kpis.get("sums_per_currency", {}).items():
        line = f"{cur}: {amt:,.2f}"
        pdf.multi_cell(0,6, line)
    pdf.ln(6)

    # Charts - new page
    pdf.add_page()
    for name, path in chart_paths.items():
        try:
            pdf.set_font("DejaVu","B",12)
        except:
            pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8, name, ln=True)
        try:
            pdf.image(path, w=160)
        except:
            pdf.multi_cell(0,6, f"(Chart {name} missing)")
        pdf.ln(6)

    return io.BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))

# -------------------------
# Email (MailerSend) helper
# -------------------------
def send_pdf_via_mailersend(api_key, sender_email, sender_name, recipient_email, subject, text_content, pdf_bytes, filename="report.pdf"):
    if not api_key:
        raise RuntimeError("MAILERSEND_API_KEY not set.")
    url = "https://api.mailersend.com/v1/email"
    # attachments require base64 encoded content
    b64 = base64.b64encode(pdf_bytes.getvalue()).decode("ascii")
    payload = {
        "from": {"email": sender_email, "name": sender_name},
        "to": [{"email": recipient_email}],
        "subject": subject,
        "content": [{"type": "text/plain", "value": text_content}],
        "attachments": [{"content": b64, "filename": filename}]
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    return resp

# -------------------------
# Main UI logic
# -------------------------
st.title("SAP Automatz — Procurement Analytics (v26.0)")

uploaded = st.file_uploader("Upload SAP PO / GRN CSV or XLSX", type=["csv", "xlsx"])
# Preserve uploaded file in session so we can regenerate AI without re-upload
if uploaded:
    # store raw bytes so we can reload
    st.session_state.upload_bytes = uploaded.getvalue()
    st.session_state.upload_name = uploaded.name

# Exchange rates UI (simple defaults)
st.sidebar.header("Options")
base_currency = st.sidebar.selectbox("Base currency for combined view", ["USD","INR","EUR","GBP","JPY","CNY"], index=0)
multicurrency_mode = st.sidebar.checkbox("Enable multi-currency conversion (show combined total)", value=True)

# show exchange rate editor for conversions (optional)
st.sidebar.markdown("**Exchange rates (1 unit currency -> USD)** (editable)")
default_rates = {"USD":1.0,"INR":0.012,"EUR":1.05,"GBP":1.25,"JPY":0.0070,"CNY":0.14}
rates = {}
for code in ["USD","INR","EUR","GBP","JPY","CNY"]:
    rates[code] = st.sidebar.number_input(f"{code} → USD", value=float(os.getenv(f"RATE_{code}", default_rates[code])), format="%.6f")

# Recipient email UI
st.sidebar.markdown("---")
st.sidebar.markdown("**Email Delivery (optional)**")
recipient_email = st.sidebar.text_input("Send PDF to email (optional):", value="")
sender_email = os.getenv("MAILER_FROM_EMAIL", "sapautomatz@gmail.com")
sender_name = os.getenv("MAILER_FROM_NAME", "SAPAutomatz")

# Re-generate AI button (uses session state to avoid reupload)
if "upload_bytes" in st.session_state and st.button("Regenerate AI Insights"):
    # reload DataFrame from saved bytes
    try:
        content = st.session_state.upload_bytes
        if st.session_state.upload_name.lower().endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
        df = normalize_columns(df)
        df = coerce_types(df)
        df_parsed, kpis = calculate_kpis_and_parse(df)
        currency_symbol = detect_currency_symbol(df)
        # prompt & AI
        currency_summary_str = ", ".join([f"{c}: {v:,.2f}" for c,v in kpis["sums_per_currency"].items()])
        prompt = build_ai_prompt(kpis, currency_summary_str)
        ai_text = get_ai_summary(prompt)
        # charts
        vendor_chart = save_vendor_chart(df_parsed, out_path="vendor_chart.png")
        material_chart = save_material_pie(df_parsed, out_path="material_chart.png")
        trend_chart = save_trend_chart(df_parsed, out_path="trend_chart.png")
        charts_dict = {k:v for k,v in [("Top Vendors", vendor_chart), ("Material Spend", material_chart), ("Monthly Trend", trend_chart)] if v}
        # store in session
        st.session_state.df = df_parsed
        st.session_state.kpis = kpis
        st.session_state.ai_text = ai_text
        st.session_state.charts = charts_dict
        st.success("✅ AI Insights regenerated.")
    except Exception as e:
        st.error(f"Regenerate error: {e}")

# If uploaded, process and show results
if "upload_bytes" in st.session_state and "df" not in st.session_state:
    try:
        content = st.session_state.upload_bytes
        name = st.session_state.upload_name
        if name.lower().endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
        df = normalize_columns(df)
        df = coerce_types(df)
        df_parsed, kpis = calculate_kpis_and_parse(df)
        currency_symbol = detect_currency_symbol(df)
        currency_summary_str = ", ".join([f"{c}: {v:,.2f}" for c,v in kpis["sums_per_currency"].items()])
        prompt = build_ai_prompt(kpis, currency_summary_str)
        ai_text = get_ai_summary(prompt)
        vendor_chart = save_vendor_chart(df_parsed, out_path="vendor_chart.png")
        material_chart = save_material_pie(df_parsed, out_path="material_chart.png")
        trend_chart = save_trend_chart(df_parsed, out_path="trend_chart.png")
        charts_dict = {k:v for k,v in [("Top Vendors", vendor_chart), ("Material Spend", material_chart), ("Monthly Trend", trend_chart)] if v}
        st.session_state.df = df_parsed
        st.session_state.kpis = kpis
        st.session_state.ai_text = ai_text
        st.session_state.charts = charts_dict
    except Exception as e:
        st.error(f"Processing error: {e}")

# Display analysis if available
if "df" in st.session_state:
    st.markdown("### Data preview (first 8 rows)")
    st.dataframe(st.session_state.df.head(8))

    st.markdown("### AI Executive Summary")
    st.write(st.session_state.ai_text or "No AI summary available.")

    st.markdown("### Charts")
    for k,v in st.session_state.charts.items():
        st.image(v, caption=k)

    # Multi-currency conversion and combined total
    if multicurrency_mode:
        sums = st.session_state.kpis.get("sums_per_currency", {})
        converted = {}
        for cur, amt in sums.items():
            # convert to USD then to base_currency
            rate_cur = rates.get(cur, default_rates.get(cur, 1.0))
            usd_value = amt * rate_cur
            # convert USD -> base
            if base_currency == "USD":
                converted[cur] = usd_value
            else:
                rate_base = rates.get(base_currency, default_rates.get(base_currency, 1.0))
                converted[cur] = usd_value / rate_base
        combined_total = sum(converted.values())
        st.markdown(f"**Combined total in {base_currency}:** {combined_total:,.2f}")

    # Generate PDF and offer download
    if st.button("Generate PDF (Preview & download)"):
        try:
            pdf_bytes = generate_pdf(st.session_state.ai_text, st.session_state.kpis, st.session_state.charts, display_currency_symbol=detect_currency_symbol(st.session_state.df))
            st.session_state.pdf_bytes = pdf_bytes  # store
            st.download_button("📄 Download PDF Report", pdf_bytes, f"SAPAutomatz_Report_{datetime.date.today()}.pdf", "application/pdf")
            st.success("PDF generated and ready for download.")
        except Exception as e:
            st.error(f"PDF generation error: {e}")

    # Send email (optional)
    if MAILERSEND_API_KEY:
        st.markdown("---")
        st.markdown("### Email delivery")
        to_email = st.text_input("Recipient email", value=recipient_email)
        email_subject = st.text_input("Email subject", value=f"SAPAutomatz Report - {datetime.date.today()}")
        email_body = st.text_area("Email body (plain text)", value="Please find the attached procurement analytics report.")
        if st.button("Send PDF by email"):
            if "pdf_bytes" not in st.session_state:
                st.error("Please generate the PDF first (click 'Generate PDF' above).")
            else:
                try:
                    resp = send_pdf_via_mailersend(MAILERSEND_API_KEY, sender_email, sender_name, to_email, email_subject, email_body, st.session_state.pdf_bytes, filename=f"SAPAutomatz_Report_{datetime.date.today()}.pdf")
                    if resp.status_code in (200,201,202):
                        st.success("✅ Email sent successfully.")
                    else:
                        st.error(f"Email failed: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Email error: {e}")
    else:
        st.info("MailerSend API key not configured in environment. Skipping email delivery option.")

else:
    st.info("Upload a SAP PO/GRN CSV or XLSX to analyze. After upload you'll be able to generate PDF, download, regenerate AI insights, and optionally email the PDF.")

# -------------------------
# End of file
# -------------------------
