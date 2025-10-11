# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI App (v24.0)
# Global Currency & Multi-Currency Comparison + Unicode-safe PDF
# ==========================================================
# Features:
# - Cross-platform font handling (Windows / Linux)
# - Auto-detect currency per-row and per-file
# - Multi-currency comparison mode with user-editable exchange rates
# - Unicode-safe PDF (DejaVu) with color-coded KPI boxes
# - Charts & Executive Summary
# ==========================================================

import os
import io
import re
import json
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

# -----------------------
# Config - update as needed
# -----------------------
BACKEND_URL = "https://sapautomatz-backend.onrender.com"  # your backend verify endpoint
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set in env
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

# Fonts: auto-detect platform (Windows -> local fonts/, Linux -> system path)
if platform.system() == "Windows":
    FONT_PATH = "./fonts/DejaVuSans.ttf"
    FONT_PATH_BOLD = "./fonts/DejaVuSans-Bold.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# Streamlit page config
# -----------------------
st.set_page_config(page_title="SAP Automatz - Procurement Analytics AI", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

# HEADER
c1, c2 = st.columns([1, 3])
with c1:
    st.image(LOGO_URL, width=160)
with c2:
    st.markdown("<h2 style='margin-bottom:0'>SAP Automatz Procurement Analytics AI</h2>"
                "<p style='color:#555;margin-top:0'>Automate. Analyze. Accelerate 🚀</p>", unsafe_allow_html=True)
st.divider()

# -----------------------
# Utility functions
# -----------------------
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
    for c in ["PO_DATE", "GRN_DATE"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["QUANTITY"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Value parsing: handles numeric, or strings with currency symbols like "₹1234" or "$1,234.56"
CURRENCY_SYMBOLS = {"₹":"INR", "$":"USD", "€":"EUR", "£":"GBP", "¥":"JPY"}  # ¥ maps to JPY or CNY - we use JPY by default

def parse_value_and_currency(val, default_currency=None):
    """Return (numeric_value, currency_code or None)"""
    if pd.isna(val):
        return (np.nan, None)
    if isinstance(val, (int, float, np.number)):
        return (float(val), default_currency)
    s = str(val).strip()
    # If explicit currency column exists, caller will pass default_currency
    # Look for currency symbols at start
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
    # If s contains symbol anywhere, find it
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in s:
            num = re.sub(r"[^\d.\-]", "", s)
            try:
                return (float(num), code)
            except:
                return (np.nan, code)
    # else try to parse as number
    try:
        return (float(s.replace(",", "")), default_currency)
    except:
        return (np.nan, default_currency)

def detect_currency_symbol_from_headers_or_sample(df):
    """Return a currency symbol (like ₹ or $) or None"""
    header = " ".join(df.columns).upper()
    # see country codes in header
    for code, sym in [("INR","₹"),("USD","$"),("EUR","€"),("GBP","£"),("JPY","¥"),("CNY","¥")]:
        if code in header:
            return sym
    # sample text
    for col in df.columns:
        if df[col].dtype == "object":
            sample = " ".join(df[col].astype(str).head(10).tolist())
            for sym in CURRENCY_SYMBOLS.keys():
                if sym in sample:
                    return sym
    return None

# -----------------------
# Exchange rates and conversion
# -----------------------
DEFAULT_EXCHANGE_RATES = {
    "USD": 1.0,      # base
    "INR": 0.012,    # 1 INR = 0.012 USD approx
    "EUR": 1.05,     # 1 EUR = 1.05 USD
    "GBP": 1.25,     # 1 GBP = 1.25 USD
    "JPY": 0.0070,   # 1 JPY = 0.007 USD
    "CNY": 0.14      # 1 CNY = 0.14 USD
}

def convert_to_base(amount, currency_code, rates, base="USD"):
    """Convert numeric amount (in currency_code) to base using rates dict (rate = 1 unit currency -> USD)"""
    if amount is None or pd.isna(amount):
        return np.nan
    if currency_code is None:
        # assume base if unknown
        currency_code = base
    if currency_code not in rates:
        # unknown currency: assume 1:1 to base
        return amount
    # rates are expressed as 1 unit currency -> USD
    # To convert any currency to base, convert to USD then to base if base != USD
    usd = amount * rates[currency_code]
    if base == "USD":
        return usd
    # if base other than USD, we'd divide by rates[base] to get base
    if rates.get(base):
        return usd / rates[base]
    return usd

# -----------------------
# KPI calculation & prompt
# -----------------------
def calculate_kpis_and_prepare(df):
    """Parse VALUE and CURRENCY columns and produce dataframe with numeric 'AMOUNT' and 'CURRENCY' columns"""
    # Ensure currency column exists; if not, we'll try to infer
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = None
    # build amounts
    amounts = []
    codes = []
    # detect default symbol in file
    file_sym = detect_currency_symbol_from_headers_or_sample(df)
    inferred_default = None
    if file_sym:
        inferred_default = CURRENCY_SYMBOLS.get(file_sym, None)
    for idx, row in df.iterrows():
        raw_val = row.get("VALUE", None)
        default_curr = row.get("CURRENCY", None) or inferred_default
        num, code = parse_value_and_currency(raw_val, default_curr)
        if code is None:
            code = default_curr
        amounts.append(num)
        codes.append(code)
    df["AMOUNT"] = amounts
    df["CURRENCY_DETECTED"] = codes
    # If there are still missing currency codes, fill with inferred_default or 'USD'
    df["CURRENCY_DETECTED"].fillna(inferred_default or "USD", inplace=True)

    # Numeric cleanup
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0.0)

    # KPIs basic
    kpis = {}
    kpis["records"] = len(df)
    # sum per currency
    sums_per_currency = df.groupby("CURRENCY_DETECTED")["AMOUNT"].sum().to_dict()
    kpis["sums_per_currency"] = sums_per_currency
    kpis["total_records"] = len(df)
    # total spend (no conversion) also
    kpis["total_spend_raw"] = df["AMOUNT"].sum()
    # If PO_DATE & GRN_DATE present compute cycle days
    if "PO_DATE" in df.columns and "GRN_DATE" in df.columns:
        try:
            df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
            df["GRN_DATE"] = pd.to_datetime(df["GRN_DATE"], errors="coerce")
            df["CYCLE_DAYS"] = (df["GRN_DATE"] - df["PO_DATE"]).dt.days
            kpis["avg_cycle_days"] = float(df["CYCLE_DAYS"].mean())
            kpis["delayed_count"] = int(df[df["CYCLE_DAYS"] > 7].shape[0])
        except Exception:
            kpis["avg_cycle_days"] = None
            kpis["delayed_count"] = 0
    else:
        kpis["avg_cycle_days"] = None
        kpis["delayed_count"] = 0

    # Top vendor by AMOUNT in their currencies (not converted)
    if "VENDOR" in df.columns:
        try:
            top_vendor = df.groupby("VENDOR")["AMOUNT"].sum().idxmax()
            kpis["top_vendor"] = top_vendor
        except Exception:
            kpis["top_vendor"] = "N/A"
    else:
        kpis["top_vendor"] = "N/A"

    return df, kpis

def build_prompt_for_ai(kpis, currency_summary_str):
    return f"""
You are a procurement analytics assistant. Use the following summary data and provide a concise executive summary.

Summary:
Total records: {kpis['records']}
Totals by currency: {currency_summary_str}
Average cycle days: {kpis.get('avg_cycle_days')}
Delayed shipments (>7 days): {kpis.get('delayed_count')}
Top vendor (by amount): {kpis.get('top_vendor')}

Provide four short sections:
1) Executive Summary
2) Key Observations
3) Root Causes
4) Actionable Recommendations

Keep responses short and professional (approx 300-450 words).
"""

def get_ai_summary(prompt):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are a senior procurement analyst producing an executive report."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
            max_tokens=700
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI generation error: {e}"

# -----------------------
# Chart helpers
# -----------------------
def save_vendor_chart(df, out_path="/tmp/vendor_chart.png"):
    if "VENDOR" in df.columns:
        top = df.groupby("VENDOR")["AMOUNT"].sum().nlargest(10)
        plt.figure(figsize=(8,4))
        top.plot(kind="bar", color="steelblue")
        plt.title("Top Vendors by Amount")
        plt.ylabel("Amount (native currency)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
    return None

def save_monthly_trend_chart(df, out_path="/tmp/trend_chart.png"):
    if "PO_DATE" in df.columns:
        df2 = df.dropna(subset=["PO_DATE"])
        if not df2.empty:
            df2["MONTH"] = df2["PO_DATE"].dt.to_period("M").astype(str)
            monthly = df2.groupby("MONTH")["AMOUNT"].sum().sort_index()
            plt.figure(figsize=(8,4))
            monthly.plot(marker='o', color='darkorange')
            plt.title("Monthly Spend Trend (native currency)")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return out_path
    return None

def save_material_pie(df, out_path="/tmp/material_chart.png"):
    if "MATERIAL" in df.columns:
        mat = df.groupby("MATERIAL")["AMOUNT"].sum().nlargest(6)
        if not mat.empty:
            plt.figure(figsize=(5,5))
            plt.pie(mat, labels=mat.index, autopct="%1.1f%%")
            plt.title("Material Spend Distribution")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return out_path
    return None

# -----------------------
# PDF generation (Unicode-safe + multi-currency summary and KPI boxes)
# -----------------------
class DejaVuPDF(FPDF):
    def header(self):
        self.set_fill_color(33,86,145)
        self.rect(0,0,210,20,'F')
        try:
            self.image(LOGO_URL, 10, 2, 25)
        except:
            pass
        self.set_text_color(255,255,255)
        self.set_font("DejaVu", "B", 14)
        self.cell(0,10,"SAP Automatz - Procurement Analytics Report", align="C", ln=True)
        self.ln(2)
    def footer(self):
        self.set_y(-15)
        # ensure font exists on each footer call
        try:
            self.set_font("DejaVu", "I", 9)
        except:
            # fallback to builtin
            self.set_font("Helvetica", "I", 9)
        self.set_text_color(130,130,130)
        self.cell(0,10,"©2025 SAP Automatz – Powered by Gen AI", align="C")

def ensure_fonts_registered(pdf):
    # Register DejaVu fonts once (safe)
    try:
        if os.path.exists(FONT_PATH):
            pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        if os.path.exists(FONT_PATH_BOLD):
            pdf.add_font("DejaVu", "B", FONT_PATH_BOLD, uni=True)
    except Exception:
        pass

def generate_pdf_with_multicurrency(ai_text, kpis, chart_paths, currency_base, exchange_rates, converted_totals):
    """Generate a PDF which is Unicode-safe, includes KPI boxes and multi-currency converted totals."""
    pdf = DejaVuPDF()
    ensure_fonts_registered(pdf)
    # set default font (safe)
    try:
        pdf.set_font("DejaVu", "", 12)
    except:
        pdf.set_font("Helvetica", "", 12)
    pdf.add_page()

    # Header line
    pdf.cell(0,10, "📈 Executive Summary Dashboard", ln=True)
    pdf.ln(6)

    # KPI boxes (color-coded)
    def kpi_color(v, good, warn):
        try:
            val = float(v)
        except:
            return (180,180,180)
        if val <= good:
            return (120,200,120)
        elif val <= warn:
            return (255,210,80)
        else:
            return (255,100,100)

    metrics = [
        (f"Total Spend (native)", kpis.get("total_spend_raw", 0), (100000, 500000)),
        (f"Avg Cycle Time (days)", kpis.get("avg_cycle_days", 0) or 0, (7, 15)),
        (f"Delayed Shipments (>7d)", kpis.get("delayed_count", 0), (10, 30))
    ]
    x0, y, w, h = 15, 40, 60, 20
    for label, val, thr in metrics:
        r,g,b = kpi_color(val, thr[0], thr[1])
        pdf.set_fill_color(r,g,b)
        pdf.rect(x0, y, w, h, "F")
        pdf.set_xy(x0+2, y+2)
        pdf.set_text_color(0,0,0)
        pdf.set_font("DejaVu", "B", 11 if os.path.exists(FONT_PATH) else 10)
        val_display = f"{val:,.0f}" if isinstance(val, (int, float, np.number)) else str(val)
        pdf.multi_cell(w-4, 6, f"{label}\n{val_display}", align="C")
        x0 += (w + 5)
    pdf.ln(35)

    # Executive summary (AI)
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0,8, "💼 Executive Summary", ln=True)
    pdf.set_font("DejaVu", "", 11)
    for line in ai_text.split("\n"):
        pdf.multi_cell(0,6, line.strip())
    pdf.ln(6)

    # Multi-currency converted totals (to base)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0,8, f"🌐 Multi-Currency Conversion (Base: {currency_base})", ln=True)
    pdf.set_font("DejaVu", "", 11)
    for cur, amt in kpis.get("sums_per_currency", {}).items():
        converted = converted_totals.get(cur, 0.0)
        pdf.multi_cell(0,6, f"{cur}: {amt:,.2f}  →  {currency_base}{converted:,.2f}")
    pdf.ln(6)

    # Charts
    pdf.add_page()
    for name, path in chart_paths.items():
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0,8, name, ln=True)
        try:
            pdf.image(path, w=160)
        except:
            pdf.multi_cell(0,6, f"(Chart {name} unavailable)")
        pdf.ln(6)

    return io.BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))

# -----------------------
# Streamlit UI
# -----------------------
st.title("SAP Automatz — Multi-currency Procurement Analytics")

uploaded = st.file_uploader("Upload SAP PO/GRN CSV or XLSX (test dataset has mixed currencies)", type=["csv", "xlsx"])
multicurrency_mode = st.checkbox("Enable Multi-currency Comparison Mode (convert to base currency)", value=True)

# editable exchange rates (user can override)
st.markdown("**Exchange rates (1 unit currency → USD)** — edit if you want different rates.")
rates = DEFAULT_EXCHANGE_RATES.copy() if 'DEFAULT_EXCHANGE_RATES' in globals() else DEFAULT_EXCHANGE_RATES
# Allow user to override
col1, col2, col3 = st.columns(3)
with col1:
    rates["USD"] = st.number_input("USD → USD", value=rates.get("USD",1.0), format="%.6f")
    rates["INR"] = st.number_input("INR → USD", value=rates.get("INR",0.012), format="%.6f")
with col2:
    rates["EUR"] = st.number_input("EUR → USD", value=rates.get("EUR",1.05), format="%.6f")
    rates["GBP"] = st.number_input("GBP → USD", value=rates.get("GBP",1.25), format="%.6f")
with col3:
    rates["JPY"] = st.number_input("JPY → USD", value=rates.get("JPY",0.0070), format="%.6f")
    rates["CNY"] = st.number_input("CNY → USD", value=rates.get("CNY",0.14), format="%.6f")

if uploaded:
    # load file
    if uploaded.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    df = normalize_columns(df)
    df = coerce_types(df)

    # parse values and currencies
    df_parsed, kpis = calculate_kpis_and_prepare(df)

    st.markdown("### Data preview")
    st.dataframe(df_parsed.head(8))

    # generate charts
    vendor_chart = save_vendor_chart(df_parsed, out_path="vendor_chart.png")
    material_chart = save_material_pie(df_parsed, out_path="material_chart.png")
    trend_chart = save_monthly_trend_chart(df_parsed, out_path="trend_chart.png")
    chart_paths = {k:v for k,v in [("Top Vendors", vendor_chart), ("Material Spend", material_chart), ("Monthly Trend", trend_chart)] if v}

    # Show totals by currency
    st.markdown("### Totals by currency (native amounts)")
    st.table(pd.DataFrame(list(kpis["sums_per_currency"].items()), columns=["Currency","Total Amount"]))

    # if multicurrency mode convert sums to selected base
    base_currency = st.selectbox("Select base currency for conversion", ["USD","INR","EUR","GBP","JPY","CNY"], index=0)
    converted_totals = {}
    if multicurrency_mode:
        # convert each currency sum to base_currency using rates
        for cur, amt in kpis["sums_per_currency"].items():
            # if cur is None or nan, assume base
            cur_code = cur or "USD"
            converted = convert_to_base(amt, cur_code, rates, base=base_currency)
            converted_totals[cur_code] = converted
        # show converted totals
        st.markdown(f"### Totals converted to {base_currency}")
        st.table(pd.DataFrame(list(converted_totals.items()), columns=[ "Currency", f"Total in {base_currency}" ]))

        # also show combined total in base currency
        combined_total = sum(converted_totals.values())
        st.metric(f"Combined Total ({base_currency})", f"{combined_total:,.2f}")

    # prepare AI prompt
    currency_summary_str = ", ".join([f"{c}: {v:,.2f}" for c,v in kpis["sums_per_currency"].items()])
    ai_prompt = build_prompt_for_ai(kpis, currency_summary_str)
    ai_text = get_ai_summary(ai_prompt)

    st.markdown("### AI Executive Summary (preview)")
    st.write(ai_text)

    # Generate PDF (include converted totals if multicurrency)
    if st.button("Generate Executive PDF Report"):
        # ensure font files available
        if not os.path.exists(FONT_PATH) or not os.path.exists(FONT_PATH_BOLD):
            st.warning("DejaVu fonts not found at expected path. For Windows, place DejaVuSans.ttf and DejaVuSans-Bold.ttf into ./fonts/. For Linux use system fonts path.")
        pdf_bytes = generate_pdf_with_multicurrency(ai_text, kpis, chart_paths, base_currency, rates, converted_totals)
        st.download_button("📄 Download PDF Report", pdf_bytes, f"SAPAutomatz_Report_{datetime.date.today()}.pdf", "application/pdf")

    st.success("Analysis complete.")

else:
    st.info("Upload a SAP PO/GRN CSV or XLSX file to analyze. Use the test generator script (below) to create a sample dataset with mixed currencies.")

# -----------------------
# End of file
# -----------------------
