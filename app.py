# app.py
# SAP Automatz – v41 (Stable release)
# Single-file production-ready Streamlit app:
# - Verify access
# - Upload CSV/XLSX
# - Compute KPIs & risk
# - Show dashboard + charts
# - Generate Unicode-safe PDF with cover, charts, summary, signature
# - No external font downloads; uses built-in Helvetica

import os
import io
import re
import datetime
import math
import urllib
from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from fpdf import FPDF

# Optional: import OpenAI if available (works if OPENAI_API_KEY is set)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Optional unidecode for safer transliteration; fallback to simple cleanup
try:
    from unidecode import unidecode
    def _unidecode(s): return unidecode(s)
except Exception:
    def _unidecode(s): return s

# ---------------- CONFIG ----------------
MODEL = "gpt-4o-mini"   # used if OpenAI available
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"
VALID_KEYS = {"SAPMM-00000000000000", "DEMO-ACCESS-12345"}  # replace with your real keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    OPENAI_AVAILABLE = False

# ---------------- STREAMLIT UI CONFIG ----------------
st.set_page_config(page_title="SAP Automatz – Procurement Analytics", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

# Header
col_logo, col_head = st.columns([1, 3])
with col_logo:
    st.image(LOGO_URL, width=140)
with col_head:
    st.markdown(
        "<h2 style='margin:0 0 4px 0;color:#1a237e'>SAP Automatz – Procurement Analytics</h2>"
        "<div style='color:#444;margin:0;font-size:13px'>Automate. Analyze. Accelerate.</div>",
        unsafe_allow_html=True
    )
st.divider()

# ---------------- SESSION STATE ----------------
if "verified" not in st.session_state:
    st.session_state.verified = False
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "df" not in st.session_state:
    st.session_state.df = None
if "kpis" not in st.session_state:
    st.session_state.kpis = None
if "risk" not in st.session_state:
    st.session_state.risk = None
if "ai_text" not in st.session_state:
    st.session_state.ai_text = ""


# ---------------- HELPERS ----------------
def sanitize_text_for_pdf(text: str) -> str:
    """
    Convert to plain ASCII-latin-1 safe string for FPDF.
    Uses unidecode if available then strips non-latin-1.
    """
    if text is None:
        return ""
    s = str(text)
    s = _unidecode(s)
    # replace common bullets and smart quotes with ascii equivalents
    s = s.replace("•", "-").replace("—", "-").replace("–", "-")
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # strip any characters outside basic latin1
    s = s.encode("latin-1", "ignore").decode("latin-1")
    # remove control characters
    s = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", " ", s)
    return s


def parse_amount_and_currency(raw_value, fallback="INR"):
    """
    Accept strings like '€ 1,234.56', 'Rs. 1,234', '1000', numeric.
    Returns (float_amount, currency_code)
    """
    if pd.isna(raw_value):
        return 0.0, fallback
    if isinstance(raw_value, (int, float, np.number)):
        return float(raw_value), fallback
    s = str(raw_value).strip()
    # Map currency symbols to codes (extend as needed)
    sym_map = {"₹": "INR", "Rs": "INR", "INR": "INR", "$": "USD", "USD": "USD", "€": "EUR", "EUR": "EUR", "GBP": "GBP"}
    detected = fallback
    for sym, code in sym_map.items():
        if sym in s:
            detected = code
            s = s.replace(sym, "")
    # remove commas and non-numeric except dot and minus
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        val = float(s) if s not in ("", ".", "-") else 0.0
    except Exception:
        val = 0.0
    return val, detected


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # try normalize column names to uppercase for easier handling
    df.columns = [c.strip() for c in df.columns]
    if "AMOUNT" not in df.columns and "AMT" in df.columns:
        df.rename(columns={"AMT": "AMOUNT"}, inplace=True)
    if "AMOUNT" not in df.columns:
        # try to guess any numeric column
        for c in df.columns:
            if df[c].dtype in (int, float) and "amount" in c.lower():
                df.rename(columns={c: "AMOUNT"}, inplace=True)
                break
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = np.nan
    # build numeric amount and detected currency
    amounts = []
    currencies = []
    for idx, row in df.iterrows():
        a, cur = parse_amount_and_currency(row.get("AMOUNT", 0), fallback=row.get("CURRENCY", np.nan) or "INR")
        amounts.append(a)
        currencies.append(cur)
    df["AMOUNT_NUM"] = amounts
    df["CURRENCY_DETECTED"] = currencies
    # columns fallback
    if "VENDOR" not in df.columns:
        # try similar
        for c in df.columns:
            if "vendor" in c.lower():
                df.rename(columns={c: "VENDOR"}, inplace=True)
                break
    if "MATERIAL" not in df.columns:
        for c in df.columns:
            if "material" in c.lower() or "mat" == c.lower():
                df.rename(columns={c: "MATERIAL"}, inplace=True)
                break
    # fill missing vendor/material
    if "VENDOR" in df.columns:
        df["VENDOR"] = df["VENDOR"].astype(str).fillna("Unknown")
    if "MATERIAL" in df.columns:
        df["MATERIAL"] = df["MATERIAL"].astype(str).fillna("Unknown")
    return df


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    df = clean_dataframe(df)
    # parse PO_DATE if exists
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict()
    total_spend = float(sum(totals.values())) if totals else 0.0
    dominant = max(totals, key=totals.get) if totals else None
    top_v = df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "VENDOR" in df.columns else {}
    if "QUANTITY" in df.columns:
        top_m = df.groupby("MATERIAL")["QUANTITY"].sum().nlargest(10).to_dict()
    else:
        top_m = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "MATERIAL" in df.columns else {}
    monthly = {}
    if "PO_DATE" in df.columns:
        d = df.dropna(subset=["PO_DATE"])
        if not d.empty:
            d["YM"] = d["PO_DATE"].dt.to_period("M").astype(str)
            monthly = d.groupby("YM")["AMOUNT_NUM"].sum().to_dict()
    return {
        "totals": totals,
        "total_spend": total_spend,
        "dominant": dominant,
        "top_v": top_v,
        "top_m": top_m,
        "monthly": monthly,
        "records": len(df),
        "df": df
    }


def compute_procurement_risk(k: Dict[str, Any]) -> Dict[str, Any]:
    df = k.get("df")
    totals = k.get("totals", {})
    total_spend = k.get("total_spend", 0.0)
    try:
        v = df.groupby("VENDOR")["AMOUNT_NUM"].sum()
        nv = v.size if not v.empty else 0
        top_share = (v.max() / total_spend) if total_spend and not v.empty else 1.0
    except Exception:
        nv = 0
        top_share = 1.0
    v_conc = max(0.0, (1.0 - top_share)) * 100
    v_div = min(100.0, (nv / 50) * 100)
    if totals and total_spend:
        dom = k.get("dominant")
        dom_share = totals.get(dom, 0.0) / total_spend if dom else 1.0
        c_expo = dom_share * 100
    else:
        c_expo = 100.0
    mvals = list(k.get("monthly", {}).values())
    if len(mvals) >= 3 and np.mean(mvals) > 0:
        cv = np.std(mvals) / (np.mean(mvals) + 1e-9)
        m_vol = max(0.0, 1 - min(cv, 2)) * 100
    else:
        m_vol = 80.0
    score = v_conc * 0.25 + v_div * 0.25 + c_expo * 0.25 + m_vol * 0.25
    score = float(max(0.0, min(100.0, score)))
    band = "Low" if score >= 67 else ("Medium" if score >= 34 else "High")
    return {
        "score": score,
        "band": band,
        "breakdown": {
            "Vendor Concentration": v_conc,
            "Vendor Diversity": v_div,
            "Currency Exposure": c_expo,
            "Monthly Volatility": m_vol
        }
    }


def generate_ai_text(k: Dict[str, Any]) -> str:
    """
    If OpenAI available and configured, use it. Otherwise return a deterministic
    summary built from the KPIs.
    """
    kpi_summary = (
        f"Total spend: {k['total_spend']:.2f}\n"
        f"Currencies: {', '.join([f'{c}:{v:.2f}' for c, v in k['totals'].items()])}\n"
        f"Top vendors: {', '.join(list(k['top_v'].keys())[:5])}\n"
    )
    if OPENAI_AVAILABLE and client:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a procurement analytics expert."},
                    {"role": "user", "content": f"Generate an executive summary and 4 recommendations based on:\n{kpi_summary}"}
                ],
                temperature=0.2, max_tokens=700
            )
            # new openai sdk returns structure choices[0].message.content
            ai_text = resp.choices[0].message.content
            return sanitize_text_for_pdf(ai_text)
        except Exception as e:
            return sanitize_text_for_pdf(f"AI failed: {e}\n\nBasic summary:\n{kpi_summary}")
    else:
        # deterministic text fallback
        lines = [
            "Executive Summary",
            f"Total spend: {k['total_spend']:.2f} ({k.get('dominant')})",
            "Top Vendors: " + (", ".join(list(k.get("top_v", {}).keys())[:5]) or "N/A"),
            "Top Materials: " + (", ".join(list(k.get("top_m", {}).keys())[:5]) or "N/A"),
            "",
            "Recommendations:",
            "1. Negotiate with top vendors for bulk discounts.",
            "2. Review inventory levels of top materials to optimize reorder points.",
            "3. Explore currency strategies for non-INR spend."
        ]
        return sanitize_text_for_pdf("\n".join(lines))


# ---------------- CHARTS ----------------
def safe_save_pie(values, labels, path, title=None):
    try:
        if not values or sum(values) == 0:
            # create placeholder
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        if title:
            ax.set_title(title)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        # fallback placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Chart error", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def safe_save_barh(labels, values, path, title=None):
    try:
        if not values or all(v == 0 for v in values):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        y = np.arange(len(labels))
        ax.barh(y, values, color="#2E7D32")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        if title:
            ax.set_title(title)
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Chart error", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def safe_save_bar(labels, values, path, title=None):
    try:
        if not values or all(v == 0 for v in values):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color="#1565C0")
        plt.xticks(rotation=45, ha="right")
        if title:
            ax.set_title(title)
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Chart error", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def safe_save_line(x, y, path, title=None):
    try:
        if not y or all(v == 0 for v in y):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y, marker="o")
        plt.xticks(rotation=45, ha="right")
        if title:
            ax.set_title(title)
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Chart error", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_gauge(score: float, path="gauge.png"):
    try:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        colors = [(1, 0.2, 0.2), (1, 0.7, 0.2), (0.2, 0.7, 0.2)]
        splits = [0, 33, 66, 100]
        for i in range(3):
            start = -np.pi + (splits[i] / 100) * np.pi
            end = -np.pi + (splits[i + 1] / 100) * np.pi
            t = np.linspace(start, end, 50)
            ax.fill_between(np.cos(t), np.sin(t), -1.2, color=colors[i], alpha=0.9)
        th = -np.pi + (score / 100) * np.pi
        x = 0.9 * math.cos(th)
        y = 0.9 * math.sin(th)
        ax.plot([0, x], [0, y], lw=4, color="k")
        ax.scatter([0], [0], color="k", s=30)
        ax.text(0, -0.1, f"{score:.0f}", ha="center", va="center", fontsize=20, fontweight="bold")
        ax.text(0, 0.35, "Procurement Risk Index", ha="center", fontsize=12, fontweight="bold", color="#333")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 0.5)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Gauge error", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ---------------- PDF generation ----------------
class PDF(FPDF):
    def header(self):
        return

    def footer(self):
        self.set_y(-15)
        # use Helvetica (available)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"SAP Automatz Confidential | Page {self.page_no()} of {{nb}}", 0, 0, "C")

    def rect_tile(self, x, y, w, h, color, title, value):
        self.set_fill_color(*color)
        self.rect(x, y, w, h, "F")
        self.set_xy(x + 3, y + 4)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(w - 6, 6, sanitize_text_for_pdf(title), ln=True)
        self.set_xy(x + 3, y + 12)
        self.set_font("Helvetica", "", 11)
        self.cell(w - 6, 6, sanitize_text_for_pdf(str(value)))


def generate_pdf_stream(ai_text: str, k: Dict[str, Any], chart_paths: list, company: str, summary_text: str, risk: Dict[str, Any]):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "", 11)

    # Cover
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, sanitize_text_for_pdf("Executive Procurement Analysis Report"), ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, sanitize_text_for_pdf(f"Prepared for: {company}"), ln=True, align="C")
    pdf.cell(0, 8, sanitize_text_for_pdf(f"Generated on: {datetime.date.today().strftime('%d %B %Y')}"), ln=True, align="C")
    pdf.ln(12)
    pdf.multi_cell(0, 7, sanitize_text_for_pdf(summary_text))
    try:
        pdf.image(LOGO_URL, x=160, y=260, w=30)
    except Exception:
        pass

    # KPI summary page (compact)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Dashboard Overview", ln=True, align="C")
    y = pdf.get_y() + 5
    pdf.rect_tile(10, y, 60, 20, (33, 150, 243), "Total Spend", f"{k['total_spend']:,.2f}")
    pdf.rect_tile(75, y, 60, 20, (76, 175, 80), "Top Vendor", next(iter(k["top_v"]), "N/A"))
    pdf.rect_tile(140, y, 60, 20, (255, 167, 38), "Currency", k.get("dominant", "N/A"))
    pdf.rect_tile(10, y + 28, 190, 20, (229, 57, 53), "Risk Index", f"{risk['score']:.0f} ({risk['band']})")

    # AI insights
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "AI-Generated Insights", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, sanitize_text_for_pdf(ai_text))

    # Risk breakdown
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Procurement Risk Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for kx, vx in risk["breakdown"].items():
        pdf.cell(0, 8, f"{sanitize_text_for_pdf(kx)}: {vx:,.2f}", ln=True)

    # Charts
    for ch in chart_paths:
        if os.path.exists(ch):
            pdf.add_page()
            title = os.path.basename(ch).replace("_", " ").replace(".png", "").title()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, sanitize_text_for_pdf(title), ln=True)
            try:
                pdf.image(ch, x=20, y=35, w=170)
            except Exception:
                pdf.multi_cell(0, 7, "Unable to embed chart image.")

    # Summary + signature
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Summary of Findings", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)
    summary_block = (
        f"• Total Spend: {k['total_spend']:,.2f} {k.get('dominant', '')}\n"
        f"• Risk Score: {risk['score']:.0f} ({risk['band']})\n"
        f"• Top Vendor: {next(iter(k['top_v']), 'N/A')}\n\n"
        f"Key Recommendations:\n{ai_text[:600]}\n\n"
        "_____________________________\n"
        "Prepared by: SAP Automatz AI Suite\n"
        "Empowering Intelligent Procurement Transformation."
    )
    pdf.multi_cell(0, 7, sanitize_text_for_pdf(summary_block))

    out = io.BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))
    out.seek(0)
    return out


# ---------------- UI FLOW ----------------

# 1) Verify Access
st.subheader("🔐 Verify Access Key")
access_col1, access_col2 = st.columns([3, 1])
with access_col1:
    access_key = st.text_input("Enter access key to continue", type="password", key="access_key_input")
with access_col2:
    if st.button("Verify"):
        if access_key.strip() in VALID_KEYS:
            st.session_state.verified = True
            st.success("Access verified — you may proceed.")
            st.experimental_rerun()
        else:
            st.error("Invalid key. Please check.")

if not st.session_state.verified:
    st.stop()

# 2) Company name and file upload
company_name = st.text_input("Company Name", value="ABC Manufacturing Pvt Ltd")

st.markdown("### 1) Upload procurement file (CSV or XLSX)")
uploaded_file = st.file_uploader("Upload SAP/ERP PO or GRN extract (CSV/XLSX)", type=["csv", "xlsx"], accept_multiple_files=False)

if uploaded_file is None:
    st.info("Upload a file to generate analytics and report.")
    st.stop()

# read file
try:
    if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.session_state.df = df
st.session_state.uploaded = True

# 3) Compute KPIs
with st.spinner("Computing KPIs..."):
    try:
        k = compute_kpis(df)
        risk = compute_procurement_risk(k)
        st.session_state.kpis = k
        st.session_state.risk = risk
    except Exception as e:
        st.error(f"Error computing KPIs: {e}")
        st.stop()

# 4) Charts & dashboard
st.markdown("## Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", k["records"])
dominant_currency = k.get("dominant") or ""
c2.metric("Total Spend", f"{k['total_spend']:,.2f} {dominant_currency}")
c3.metric("Top Vendor", next(iter(k["top_v"]), "N/A"))
c4.metric("Risk", f"{risk['score']:.0f} ({risk['band']})")

# Charts generation
chart_paths = []
# gauge
gauge_path = "chart_gauge.png"
plot_gauge(risk["score"], gauge_path)
chart_paths.append(gauge_path)
# currency pie
if k["totals"]:
    vals = list(k["totals"].values())
    labs = list(k["totals"].keys())
else:
    vals, labs = [], []
cur_pie = "chart_currency.png"
safe_save_pie(vals, labs, cur_pie, "Currency Distribution")
chart_paths.append(cur_pie)
# top vendors
top_v = k.get("top_v", {})
top_v_bar = "chart_vendors.png"
if top_v:
    safe_save_barh(list(top_v.keys()), list(top_v.values()), top_v_bar, "Top Vendors by Spend")
else:
    safe_save_barh([], [], top_v_bar, "Top Vendors by Spend")
chart_paths.append(top_v_bar)
# top materials
top_m = k.get("top_m", {})
top_m_bar = "chart_materials.png"
if top_m:
    safe_save_bar(list(top_m.keys()), list(top_m.values()), top_m_bar, "Top Materials")
else:
    safe_save_bar([], [], top_m_bar, "Top Materials")
chart_paths.append(top_m_bar)
# monthly trend
monthly = k.get("monthly", {})
monthly_path = "chart_monthly.png"
if monthly:
    safe_save_line(list(monthly.keys()), list(monthly.values()), monthly_path, "Monthly Purchase Trend")
else:
    safe_save_line([], [], monthly_path, "Monthly Purchase Trend")
chart_paths.append(monthly_path)

# Display charts on screen
st.markdown("### Visual Highlights")
cols = st.columns(2)
with cols[0]:
    st.image(gauge_path, use_column_width=True, caption="Procurement Risk Index")
    st.image(cur_pie, use_column_width=True, caption="Currency Distribution")
with cols[1]:
    st.image(top_v_bar, use_column_width=True, caption="Top Vendors")
    st.image(top_m_bar, use_column_width=True, caption="Top Materials")
st.image(monthly_path, use_column_width=True, caption="Monthly Purchase Trend")

# 5) AI insights (either from OpenAI or fallback)
with st.spinner("Generating AI insights..."):
    ai_text = generate_ai_text(k)
    st.session_state.ai_text = ai_text

st.subheader("AI Executive Insights")
st.markdown(ai_text.replace("\n", "  \n"))

# 6) Generate Report button (creates PDF and offers download)
st.markdown("## Generate PDF Report")
if st.button("Generate & Preview PDF"):
    with st.spinner("Building PDF..."):
        try:
            pdf_stream = generate_pdf_stream(ai_text, k, chart_paths, company_name, ai_text[:1000], risk)
            st.success("PDF ready — click to download.")
            st.download_button("📄 Download Executive Report (PDF)", data=pdf_stream,
                               file_name="SAP_Automatz_Executive_Report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
            # show simple fallback text file
            st.download_button("Download basic summary (txt)", data=ai_text, file_name="summary.txt")

# End of app
