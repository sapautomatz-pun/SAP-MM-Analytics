# app.py
# SAP Automatz – Procurement Analytics v42_production_final (patched for robustness)

import os
import io
import re
import datetime
import tempfile
import traceback

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# Optional: OpenAI + Unicode
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from unidecode import unidecode
    def clean_text(s): return unidecode(str(s))
except Exception:
    def clean_text(s): return str(s)

# ---------------- CONFIG ----------------
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"
VALID_KEYS = {"SAPMM-00000000000000", "DEMO-ACCESS-12345"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    OPENAI_AVAILABLE = False


# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="SAP Automatz – Procurement Analytics", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 3])
with col_logo:
    st.image(LOGO_URL, width=120)
with col_title:
    st.markdown(
        "<h2 style='color:#1a237e;margin-bottom:0'>SAP Automatz – Procurement Analytics</h2>"
        "<p style='margin-top:0;color:#555;'>Automate. Analyze. Accelerate.</p>",
        unsafe_allow_html=True,
    )
st.divider()

# ensure verified default exists
st.session_state.setdefault("verified", False)

# ---------------- HELPERS ----------------
def sanitize_text_for_pdf(text):
    if not text:
        return ""
    s = clean_text(str(text))
    s = s.replace("•", "-").replace("—", "-").replace("–", "-")
    return s.encode("latin-1", "ignore").decode("latin-1")


def parse_amount_and_currency(value, fallback="INR"):
    sym_map = {"₹": "INR", "Rs": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
    if pd.isna(value):
        return 0.0, fallback
    s = str(value)
    detected = fallback
    for sym, code in sym_map.items():
        if sym in s:
            detected = code
            s = s.replace(sym, "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        val = float(s) if s not in ("", ".", "-", "-.") else 0.0
    except Exception:
        val = 0.0
    return val, detected


def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    # normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    # find amount-like column
    if "AMOUNT" not in df.columns:
        for c in df.columns:
            if "AMT" in c.upper():
                df.rename(columns={c: "AMOUNT"}, inplace=True)
                break
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"
    amounts, currencies = [], []
    for _, row in df.iterrows():
        a, c = parse_amount_and_currency(row.get("AMOUNT", 0), row.get("CURRENCY", "INR"))
        amounts.append(a)
        currencies.append(c)
    df["AMOUNT_NUM"] = amounts
    df["CURRENCY_DETECTED"] = currencies
    return df


def compute_kpis(df):
    df = prepare_dataframe(df)
    # sum by currency
    if "CURRENCY_DETECTED" in df.columns and not df["AMOUNT_NUM"].isna().all():
        totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict()
    else:
        totals = {}
    total_spend = sum(totals.values()) if totals else df["AMOUNT_NUM"].sum() if "AMOUNT_NUM" in df.columns else 0.0
    dominant = max(totals, key=totals.get) if totals else "INR"
    top_v = df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "VENDOR" in df.columns and not df["VENDOR"].isna().all() else {}
    top_m = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "MATERIAL" in df.columns and not df["MATERIAL"].isna().all() else {}
    monthly = {}
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
        temp = df.dropna(subset=["PO_DATE"])
        if not temp.empty:
            temp["MONTH"] = temp["PO_DATE"].dt.to_period("M").astype(str)
            monthly = temp.groupby("MONTH")["AMOUNT_NUM"].sum().to_dict()
    return {"totals": totals, "total_spend": float(total_spend), "dominant": dominant,
            "top_v": top_v, "top_m": top_m, "monthly": monthly, "records": len(df), "df": df}


def compute_risk(k):
    df = k.get("df", pd.DataFrame())
    total_spend = k.get("total_spend", 0.0) or 0.0
    # safe vendor series
    if "VENDOR" in df.columns:
        v = df.groupby("VENDOR")["AMOUNT_NUM"].sum()
    else:
        v = pd.Series(dtype=float)

    # if no spend at all, return neutral/clear risk to avoid divide-by-zero
    if total_spend == 0 or df.empty:
        return {"score": 0.0, "band": "Low", "breakdown": {
            "Vendor Concentration": 0.0,
            "Vendor Diversity": 0.0,
            "Currency Exposure": 0.0,
            "Monthly Volatility": 0.0
        }}

    nv = len(v) if not v.empty else 0
    if not v.empty:
        top_share = float(v.max()) / total_spend if total_spend else 1.0
    else:
        top_share = 1.0  # single vendor or no vendor detail => conservative default
    v_conc = (1 - top_share) * 100
    v_div = min(100.0, (nv / 50.0) * 100.0)
    dom = k.get("dominant", "INR")
    c_expo = (k["totals"].get(dom, 0.0) / total_spend) * 100.0 if total_spend else 100.0
    mvals = list(k.get("monthly", {}).values())
    m_vol = 100.0 * (1 - np.std(mvals) / (np.mean(mvals) + 1e-9)) if len(mvals) > 2 and np.mean(mvals) != 0 else 80.0
    # compute score robustly constrained between 0 and 100
    score = float(np.clip((v_conc + v_div + c_expo + m_vol) / 4.0, 0.0, 100.0))
    band = "Low" if score >= 67 else ("Medium" if score >= 34 else "High")
    return {"score": score, "band": band, "breakdown": {
        "Vendor Concentration": v_conc,
        "Vendor Diversity": v_div,
        "Currency Exposure": c_expo,
        "Monthly Volatility": m_vol
    }}


def generate_ai_text(k):
    base_summary = f"Total spend: {k.get('total_spend', 0.0):.2f}, Dominant currency: {k.get('dominant','INR')}, Top vendors: {list(k.get('top_v',{}).keys())[:3]}"
    if OPENAI_AVAILABLE and client:
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a procurement analytics expert."},
                    {"role": "user", "content": f"Generate an executive summary and recommendations for:\n{base_summary}"}
                ],
                temperature=0.3,
                max_tokens=400
            )
            # new OpenAI SDKs may vary in response shape; guard with try/except
            try:
                text = r.choices[0].message.content
            except Exception:
                # fallback: attempt common alt path
                text = getattr(r.choices[0], "text", "")
            return sanitize_text_for_pdf(text)
        except Exception:
            # log quietly, fall back to default summary
            pass
    # fallback summary
    return sanitize_text_for_pdf(
        f"Executive Insights:\n- Total spend: {k.get('total_spend',0.0):.2f} ({k.get('dominant','INR')})\n"
        f"- Key vendors: {', '.join(list(k.get('top_v',{}).keys())[:3]) or 'N/A'}\n"
        "Recommendations:\n- Negotiate better rates with top vendors.\n"
        "- Optimize inventory for high-demand materials.\n"
        "- Review currency exposure and diversify vendor base."
    )


# ---------------- PDF GENERATION ----------------
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"SAP Automatz | Page {self.page_no()}", 0, 0, "C")


def generate_pdf(ai_text, k, charts, company, risk):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Procurement Analytics Report", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Company: {company}", ln=True, align="C")
    pdf.cell(0, 10, f"Date: {datetime.date.today().strftime('%d-%b-%Y')}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 7, ai_text)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Risk Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, f"Risk Score: {risk.get('score',0.0):.0f} ({risk.get('band','N/A')})")
    # add charts if exist and accessible
    for ch in charts:
        try:
            if ch and os.path.exists(ch):
                pdf.add_page()
                pdf.image(ch, x=20, y=30, w=170)
        except Exception:
            # skip any chart that fails to render
            continue
    out_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
    out = io.BytesIO(out_bytes)
    out.seek(0)
    return out


# ---------------- APP FLOW ----------------
st.subheader("🔐 Verify Access Key")
col1, col2 = st.columns([3, 1])
with col1:
    access_key = st.text_input("Enter access key", type="password")
with col2:
    if st.button("Verify"):
        if access_key and access_key.strip() in VALID_KEYS:
            st.session_state["verified"] = True
            st.success("Access verified successfully.")
            st.rerun()
        else:
            st.error("Invalid key. Please try again.")

if not st.session_state.get("verified"):
    st.stop()

st.markdown("### Upload Procurement File")
file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if not file:
    st.info("Upload your procurement extract to continue.")
    st.stop()

# wrap pipeline in try/except to capture unexpected errors
try:
    # read file safely
    if file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        # CSV
        df = pd.read_csv(file)

    k = compute_kpis(df)
    risk = compute_risk(k)
    ai_text = generate_ai_text(k)

    # ---------------- CHARTS ----------------
    charts = []

    # Currency pie chart only if we have positive totals
    totals = k.get("totals", {})
    positive_vals = [v for v in totals.values() if v and v > 0]
    if totals and len(positive_vals) > 0:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(6, 4))
                labels = list(totals.keys())
                sizes = [totals[k_] for k_ in labels]
                plt.pie(sizes, labels=labels, autopct="%1.1f%%")
                plt.title("Currency Distribution")
                plt.tight_layout()
                plt.savefig(tmp.name, bbox_inches="tight")
                charts.append(tmp.name)
            plt.close()
        except Exception:
            plt.close()

    # Top vendors bar chart
    top_v = k.get("top_v", {})
    if top_v:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(7, 4))
                plt.bar(list(top_v.keys()), list(top_v.values()))
                plt.xticks(rotation=45, ha="right")
                plt.title("Top Vendors")
                plt.tight_layout()
                plt.savefig(tmp.name, bbox_inches="tight")
                charts.append(tmp.name)
            plt.close()
        except Exception:
            plt.close()

    # Top materials bar chart
    top_m = k.get("top_m", {})
    if top_m:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(7, 4))
                plt.bar(list(top_m.keys()), list(top_m.values()))
                plt.xticks(rotation=45, ha="right")
                plt.title("Top Materials")
                plt.tight_layout()
                plt.savefig(tmp.name, bbox_inches="tight")
                charts.append(tmp.name)
            plt.close()
        except Exception:
            plt.close()

    # ---------------- DISPLAY ----------------
    st.markdown("## Executive Dashboard")
    st.metric("Total Spend", f"{k.get('total_spend',0.0):,.2f} {k.get('dominant','INR')}")
    st.metric("Risk Score", f"{risk.get('score',0.0):.0f} ({risk.get('band','N/A')})")
    st.markdown("### AI Insights")
    st.write(ai_text)

    if charts:
        st.image(charts, use_container_width=True)
    else:
        st.info("Not enough data to generate charts.")

    if st.button("📄 Generate PDF Report"):
        try:
            pdf = generate_pdf(ai_text, k, charts, "ABC Manufacturing Pvt Ltd", risk)
            st.download_button("Download Report", pdf, file_name="SAP_Automatz_Report.pdf", mime="application/pdf")
        except Exception as e:
            st.error("Failed to generate PDF. See details below.")
            st.text(traceback.format_exc())

except Exception:
    st.error("An error occurred while processing the file. See details below.")
    st.text(traceback.format_exc())
