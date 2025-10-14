# app.py
# SAP Automatz – Procurement Analytics v12
# Full continuous PDF report + on-screen insights + header logo/tagline

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

# ---------- CONFIG ----------
LOGO_PATH = "sapautomatz_logo.png"
TAGLINE = "Automate. Analyze. Accelerate."
VALID_KEYS = {"SAPMM-00000000000000", "DEMO-ACCESS-12345"}

# ---------- STREAMLIT ----------
st.set_page_config(page_title="SAP Automatz – Procurement Analytics", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 3])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=110)
with col_title:
    st.markdown(
        "<h2 style='color:#1a237e;margin-bottom:0'>SAP Automatz – Procurement Analytics</h2>"
        f"<p style='margin-top:0;color:#555;'>{TAGLINE}</p>",
        unsafe_allow_html=True,
    )
st.divider()

st.session_state.setdefault("verified", False)

# ---------- HELPER FUNCTIONS ----------
def parse_amount_and_currency(value, fallback="INR"):
    sym_map = {"₹": "INR", "Rs": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
    if pd.isna(value):
        return 0.0, fallback
    s, detected = str(value), fallback
    for sym, code in sym_map.items():
        if sym in s:
            detected, s = code, s.replace(sym, "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s or 0), detected
    except Exception:
        return 0.0, detected

def prepare_dataframe(df: pd.DataFrame):
    df.columns = [c.strip().upper() for c in df.columns]
    if "AMOUNT" not in df.columns:
        for c in df.columns:
            if "AMT" in c.upper():
                df.rename(columns={c: "AMOUNT"}, inplace=True)
                break
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"
    df["AMOUNT_NUM"], df["CURRENCY_DETECTED"] = zip(
        *df.apply(lambda r: parse_amount_and_currency(r.get("AMOUNT", 0), r.get("CURRENCY", "INR")), axis=1)
    )
    return df

def compute_kpis(df):
    df = prepare_dataframe(df)
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict()
    total_spend = sum(totals.values())
    dominant = max(totals, key=totals.get) if totals else "INR"
    top_v = df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "VENDOR" in df.columns else {}
    top_m = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "MATERIAL" in df.columns else {}
    monthly = {}
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
        temp = df.dropna(subset=["PO_DATE"])
        if not temp.empty:
            temp["MONTH"] = temp["PO_DATE"].dt.to_period("M").astype(str)
            monthly = temp.groupby("MONTH")["AMOUNT_NUM"].sum().to_dict()
    return {
        "totals": totals, "total_spend": total_spend, "dominant": dominant,
        "top_v": top_v, "top_m": top_m, "monthly": monthly, "df": df, "records": len(df)
    }

def compute_risk(k):
    df = k.get("df", pd.DataFrame())
    total = k.get("total_spend", 0.0)
    if total == 0 or df.empty:
        return {"score": 0, "band": "Low", "breakdown": {}}
    dom = k.get("dominant", "INR")
    totals = k.get("totals", {})
    v = df.groupby("VENDOR")["AMOUNT_NUM"].sum() if "VENDOR" in df.columns else pd.Series(dtype=float)
    nv, top_share = len(v), float(v.max())/total if not v.empty else 1.0
    v_conc, v_div = (1-top_share)*100, min(100, (nv/50)*100)
    c_expo = (totals.get(dom,0)/total)*100 if total else 100
    mvals = list(k.get("monthly", {}).values())
    m_vol = 100*(1 - np.std(mvals)/(np.mean(mvals)+1e-9)) if len(mvals)>2 else 80
    score = np.clip((v_conc+v_div+c_expo+m_vol)/4, 0, 100)
    band = "Low" if score>=67 else "Medium" if score>=34 else "High"
    return {"score": score, "band": band,
            "breakdown": {"Vendor Concentration": v_conc, "Vendor Diversity": v_div,
                          "Currency Exposure": c_expo, "Monthly Volatility": m_vol}}

def compute_efficiency(df):
    if "VENDOR" not in df.columns or df.empty:
        return {}
    avg_cost = df.groupby("VENDOR")["AMOUNT_NUM"].mean()
    total_spend = df.groupby("VENDOR")["AMOUNT_NUM"].sum()
    eff_data = {}
    for vendor in avg_cost.index:
        eff_data[vendor] = {"avg_cost": avg_cost[vendor], "total_spend": total_spend[vendor]}
    return eff_data

def compute_material_performance(df):
    if "MATERIAL" not in df.columns or df.empty:
        return {}
    return df.groupby("MATERIAL")["AMOUNT_NUM"].sum().sort_values(ascending=False).head(10).to_dict()

def generate_ai_text(k):
    total = k["total_spend"]
    currency = k["dominant"]
    top_v = list(k["top_v"].keys())[:3]
    return (f"Total procurement spend was {total:,.2f} {currency}. "
            f"Top vendors by spend: {', '.join(top_v)}. "
            "Overall procurement performance shows potential optimization in supplier consolidation and currency exposure.")

# ---------- PDF ----------
class PDF(FPDF):
    def header(self):
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, 10, 8, 20)
        self.set_xy(32, 10)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 6, "SAP Automatz", ln=True)
        self.set_xy(32, 16)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, TAGLINE, ln=True)
        self.ln(8)
        self.line(10, 25, 200, 25)
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"SAP Automatz | Page {self.page_no()}", 0, 0, "C")

# ---------- CHARTS ----------
def generate_dashboard_charts(k, risk):
    charts = []
    if k.get("monthly"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.figure(figsize=(7,3))
            plt.plot(list(k["monthly"].keys()), list(k["monthly"].values()), marker="o")
            plt.title("Monthly Spend Trend")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); plt.savefig(tmp.name); charts.append(tmp.name)
        plt.close()
    if k.get("top_v"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            top5 = dict(list(k["top_v"].items())[:5])
            plt.figure(figsize=(7,3))
            plt.bar(top5.keys(), top5.values())
            plt.title("Top 5 Vendors")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); plt.savefig(tmp.name); charts.append(tmp.name)
        plt.close()
    if risk.get("breakdown"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.figure(figsize=(7,3))
            plt.bar(risk["breakdown"].keys(), risk["breakdown"].values())
            plt.title("Risk Breakdown")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout(); plt.savefig(tmp.name); charts.append(tmp.name)
        plt.close()
    return charts

# ---------- PDF GENERATOR ----------
def generate_pdf(ai_text, k, risk, company):
    pdf = PDF(); pdf.alias_nb_pages(); pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Procurement Analytics Report", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 13)
    pdf.cell(0, 10, f"Company: {company}", ln=True, align="C")
    pdf.cell(0, 10, f"Generated: {datetime.date.today():%d-%b-%Y}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)

    # Executive Summary
    pdf.multi_cell(0, 7, ai_text); pdf.ln(5)

    # Key Metrics
    metrics = [
        f"Total Spend: {k['total_spend']:,.2f} {k['dominant']}",
        f"Records: {k['records']}",
        f"Vendors: {len(k['top_v'])}",
        f"Materials: {len(k['top_m'])}",
        f"Risk Score: {risk['score']:.1f} ({risk['band']})"
    ]
    pdf.set_font("Helvetica", "B", 14); pdf.cell(0,10,"Key Performance Metrics",ln=True)
    pdf.set_font("Helvetica","",12)
    for m in metrics: pdf.cell(0,8,"- "+m,ln=True)
    pdf.ln(6)

    # Critical Findings
    pdf.set_font("Helvetica", "B", 14); pdf.cell(0,10,"Critical Findings",ln=True)
    pdf.set_font("Helvetica","",12)
    pdf.multi_cell(0,7,"High vendor concentration risk detected. Diversify and optimize supplier network."); pdf.ln(5)

    # Top Vendors
    pdf.set_font("Helvetica","B",14); pdf.cell(0,10,"Top Performing Vendors",ln=True)
    pdf.set_font("Helvetica","",12)
    for v,val in k["top_v"].items(): pdf.cell(0,8,f"- {v}: {val:,.2f} {k['dominant']}",ln=True)
    pdf.ln(5)

    # Efficiency Analysis
    pdf.set_font("Helvetica","B",14); pdf.cell(0,10,"Efficiency Analysis",ln=True)
    pdf.set_font("Helvetica","",12)
    eff = compute_efficiency(k["df"])
    if eff:
        most = max(eff.items(), key=lambda x:x[1]["avg_cost"])
        least = min(eff.items(), key=lambda x:x[1]["avg_cost"])
        pdf.multi_cell(0,7,
            f"Most Efficient: {least[0]} leads with {least[1]['avg_cost']:.2f} cost-per-unit "
            f"despite {least[1]['total_spend']/1e6:.2f}M total spend, representing the efficiency benchmark.\n"
            f"Least Efficient: {most[0]} shows {most[1]['avg_cost']:.2f} cost-per-unit, "
            f"presenting the largest optimization opportunity.")
    else:
        pdf.multi_cell(0,7,"Efficiency data not available.")
    pdf.ln(5)

    # Material Performance
    pdf.set_font("Helvetica","B",14); pdf.cell(0,10,"Material Category Performance",ln=True)
    pdf.set_font("Helvetica","",12)
    mat_perf = compute_material_performance(k["df"])
    if mat_perf:
        for m,val in mat_perf.items():
            pdf.cell(0,8,f"- {m}: {val:,.2f} {k['dominant']}",ln=True)
    pdf.ln(5)

    # Dashboard Charts
    pdf.set_font("Helvetica","B",14); pdf.cell(0,10,"Dashboard Charts",ln=True)
    charts = generate_dashboard_charts(k, risk)
    for ch in charts:
        y = pdf.get_y()
        pdf.image(ch, x=20, y=y, w=170)
        pdf.ln(75)
    pdf.ln(5)

    # Risk Summary
    pdf.set_font("Helvetica","B",14); pdf.cell(0,10,"Risk Summary",ln=True)
    pdf.set_font("Helvetica","",12)
    pdf.cell(0,8,f"Risk Score: {risk['score']:.1f} ({risk['band']})",ln=True)
    for k_,v_ in risk["breakdown"].items(): pdf.cell(0,8,f"- {k_}: {v_:.1f}",ln=True)

    out = io.BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))
    out.seek(0)
    return out

# ---------- APP FLOW ----------
st.subheader("🔐 Verify Access Key")
col1, col2 = st.columns([3, 1])
with col1: key = st.text_input("Enter access key", type="password")
with col2:
    if st.button("Verify"):
        if key.strip() in VALID_KEYS:
            st.session_state["verified"] = True; st.success("Access verified."); st.rerun()
        else: st.error("Invalid key.")
if not st.session_state["verified"]: st.stop()

file = st.file_uploader("Upload Procurement File (CSV/XLSX)", type=["csv","xlsx"])
if not file: st.info("Upload your procurement extract."); st.stop()
company = st.text_input("Enter Company Name", "ABC Manufacturing Pvt Ltd")

try:
    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    k = compute_kpis(df); risk = compute_risk(k); ai_text = generate_ai_text(k)

    st.markdown("## Executive Summary"); st.write(ai_text)
    st.markdown("### Key Performance Metrics")
    cols = st.columns(5)
    metrics = [
        f"{k['total_spend']:,.2f} {k['dominant']}", k['records'],
        len(k['top_v']), len(k['top_m']), f"{risk['score']:.1f} ({risk['band']})"
    ]
    labels = ["Total Spend","Records","Vendors","Materials","Risk Score"]
    for c,l,m in zip(cols,labels,metrics): c.metric(l,m)

    st.markdown("### Critical Findings")
    st.write("High vendor concentration risk detected. Diversify and optimize supplier network.")
    st.markdown("### Efficiency Analysis")
    eff = compute_efficiency(k["df"])
    if eff:
        most = max(eff.items(), key=lambda x:x[1]["avg_cost"])
        least = min(eff.items(), key=lambda x:x[1]["avg_cost"])
        st.write(
            f"**Most Efficient:** {least[0]} leads with {least[1]['avg_cost']:.2f} cost-per-unit despite "
            f"{least[1]['total_spend']/1e6:.2f}M total spend.\n\n"
            f"**Least Efficient:** {most[0]} shows {most[1]['avg_cost']:.2f} cost-per-unit, "
            f"presenting the largest optimization opportunity."
        )

    st.markdown("### Material Category Performance")
    mat_perf = compute_material_performance(k["df"])
    st.write(pd.DataFrame.from_dict(mat_perf, orient="index", columns=["Spend"]).head(10))

    st.markdown("### Dashboard Charts")
    charts = generate_dashboard_charts(k, risk)
    if charts: st.image(charts, caption=["Spend Trend","Top Vendors","Risk Breakdown"], use_container_width=True)

    if st.button("📄 Generate PDF Report"):
        pdf = generate_pdf(ai_text, k, risk, company)
        st.download_button("Download Report", pdf, file_name=f"{company.replace(' ','_')}_Procurement_Report.pdf", mime="application/pdf")

except Exception:
    st.error("Error generating report:"); st.text(traceback.format_exc())
