# app.py
# SAP Automatz – Procurement Analytics v13
# Full: mixed-currency insights (concise), AI tag, header-only logo, PDF + on-screen parity

import os
import io
import re
import datetime
import tempfile
import traceback
import math
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
AI_TAGLINE = "Insights generated automatically by SAP Automatz AI Engine"
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

# ---------- HELPERS ----------
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
        return float(s) if s not in ("", ".", "-", "-.") else 0.0, detected
    except Exception:
        return 0.0, detected

def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    # canonicalize AMOUNT column
    if "AMOUNT" not in df.columns:
        for c in df.columns:
            if "AMT" in c.upper():
                df.rename(columns={c: "AMOUNT"}, inplace=True)
                break
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"
    amounts = []
    currencies = []
    for _, r in df.iterrows():
        a, c = parse_amount_and_currency(r.get("AMOUNT", 0), r.get("CURRENCY", "INR"))
        amounts.append(a); currencies.append(c)
    df["AMOUNT_NUM"] = amounts
    df["CURRENCY_DETECTED"] = currencies
    return df

def compute_kpis(df):
    df = prepare_dataframe(df)
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict() if "CURRENCY_DETECTED" in df.columns else {}
    total_spend = float(sum(totals.values())) if totals else float(df["AMOUNT_NUM"].sum()) if "AMOUNT_NUM" in df.columns else 0.0
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
    return {"totals": totals, "total_spend": total_spend, "dominant": dominant,
            "top_v": top_v, "top_m": top_m, "monthly": monthly, "df": df, "records": len(df)}

def compute_risk(k):
    df = k.get("df", pd.DataFrame())
    total = float(k.get("total_spend", 0.0) or 0.0)
    if total == 0 or df.empty:
        return {"score": 0.0, "band": "Low", "breakdown": {}}
    dom = k.get("dominant", "INR")
    totals = k.get("totals", {})
    v = df.groupby("VENDOR")["AMOUNT_NUM"].sum() if "VENDOR" in df.columns else pd.Series(dtype=float)
    nv = len(v) if not v.empty else 0
    top_share = float(v.max()) / total if not v.empty else 1.0
    v_conc = (1 - top_share) * 100.0
    v_div = min(100.0, (nv / 50.0) * 100.0) if nv > 0 else 0.0
    c_expo = (totals.get(dom, 0.0) / total) * 100.0 if total else 100.0
    mvals = list(k.get("monthly", {}).values())
    m_vol = 100.0 * (1 - np.std(mvals) / (np.mean(mvals) + 1e-9)) if len(mvals) > 2 and np.mean(mvals) != 0 else 80.0
    score = float(np.clip((v_conc + v_div + c_expo + m_vol) / 4.0, 0.0, 100.0))
    band = "Low" if score >= 67 else ("Medium" if score >= 34 else "High")
    return {"score": score, "band": band, "breakdown": {
        "Vendor Concentration": v_conc, "Vendor Diversity": v_div,
        "Currency Exposure": c_expo, "Monthly Volatility": m_vol
    }}

def compute_efficiency_summary(df):
    # returns dict: vendor -> {"avg_cost", "total_spend", "units_est"}
    if "VENDOR" not in df.columns or df.empty:
        return {}
    avg_cost = df.groupby("VENDOR")["AMOUNT_NUM"].mean()
    total_spend = df.groupby("VENDOR")["AMOUNT_NUM"].sum()
    eff = {}
    for v in avg_cost.index:
        ac = float(avg_cost.loc[v])
        ts = float(total_spend.loc[v])
        # infer units = total_spend / avg_cost (guard)
        units = (ts / ac) if ac and ac != 0 else 0.0
        eff[v] = {"avg_cost": ac, "total_spend": ts, "units_est": units}
    return eff

def compute_material_performance(df):
    if "MATERIAL" not in df.columns or df.empty:
        return {}
    return df.groupby("MATERIAL")["AMOUNT_NUM"].sum().sort_values(ascending=False).to_dict()

# ---------- Insights (concise; mixed-currency aware) ----------
def spend_trend_insight(monthly):
    if not monthly:
        return "Spend trend: Not enough monthly data."
    months = sorted(monthly.keys())
    if len(months) < 2:
        return "Spend trend: Insufficient data to compute trend."
    last = monthly[months[-1]]
    prev = monthly[months[-2]]
    pct = ((last - prev) / (prev + 1e-9)) * 100.0
    direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
    return f"Spend trend: {direction} {abs(pct):.1f}% vs prior month."

def vendor_dependency_insight(top_v, total):
    if not top_v or total == 0:
        return "Vendor dependency: Insufficient vendor data."
    top3 = sum(list(top_v.values())[:3])
    pct = (top3 / (total + 1e-9)) * 100.0
    return f"Vendor dependency: Top 3 vendors account for {pct:.1f}% of total spend."

def currency_exposure_insight(totals):
    if not totals:
        return "Currency exposure: No currency data."
    total = sum(totals.values())
    sorted_c = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    dom, dom_amt = sorted_c[0]
    others = [(k, v) for k, v in sorted_c[1:]]
    if not others:
        return f"Currency exposure: All spend in {dom}."
    other_pct = (sum(v for _, v in others) / (total + 1e-9)) * 100.0
    other_list = ", ".join([k for k, _ in others[:3]])
    return f"Currency exposure: {dom} dominant ({dom_amt/ (total+1e-9) *100:.1f}%), {other_pct:.1f}% across {other_list}."

def efficiency_insight(eff_summary):
    if not eff_summary:
        return "Efficiency: No vendor efficiency data."
    # most efficient = lowest avg_cost; least efficient = highest avg_cost
    sorted_v = sorted(eff_summary.items(), key=lambda x: x[1]["avg_cost"])
    most_v, most_data = sorted_v[0]
    least_v, least_data = sorted_v[-1]
    # estimate annual savings if least matched most: units_est * (least_avg - most_avg)
    units = least_data["units_est"]
    gap = least_data["avg_cost"] - most_data["avg_cost"]
    est_savings = units * gap if units > 0 and gap > 0 else 0.0
    # format
    return (f"Efficiency: Most Efficient: {most_v} at {most_data['avg_cost']:.2f} cost/unit. "
            f"Least Efficient: {least_v} at {least_data['avg_cost']:.2f} cost/unit; "
            f"estimated annual saving if optimized: {est_savings:,.0f} (in vendor currency).")

def material_mix_insight(mat_perf, total):
    if not mat_perf or total == 0:
        return "Material mix: Not enough data."
    items = list(mat_perf.items())[:2]
    parts = []
    for m, v in items:
        parts.append(f"{m} ({v/ (total+1e-9) *100:.1f}%)")
    return "Material mix: Top materials - " + ", ".join(parts) + "."

def current_month_snapshot(monthly, top_v):
    if not monthly:
        return "Current month snapshot: No monthly data."
    months = sorted(monthly.keys())
    last = months[-1]
    last_val = monthly[last]
    prev = monthly[months[-2]] if len(months) >= 2 else None
    pct = ((last_val - prev) / (prev + 1e-9) *100) if prev is not None else 0.0
    top_vendor = list(top_v.keys())[0] if top_v else "N/A"
    trend = "up" if pct > 0 else "down" if pct < 0 else "flat"
    return f"Current month ({last}): {last_val:,.0f}; top vendor: {top_vendor}; trend: {trend} {abs(pct):.1f}% vs prior month."
def generate_ai_text(k):
    total = k.get("total_spend", 0.0)
    currency = k.get("dominant", "INR")
    top_v = list(k.get("top_v", {}).keys())[:3]
    if not top_v:
        top_v_text = "no major vendors identified"
    else:
        top_v_text = ", ".join(top_v)
    totals = k.get("totals", {})
    if len(totals) > 1:
        other_currencies = [c for c in totals.keys() if c != currency]
        exposure = sum(v for c, v in totals.items() if c != currency)
        exposure_pct = (exposure / (total + 1e-9)) * 100.0
        exposure_text = f" with approximately {exposure_pct:.1f}% exposure in {', '.join(other_currencies[:3])}"
    else:
        exposure_text = ""
    return (
        f"Total procurement spend was {total:,.2f} {currency}{exposure_text}. "
        f"Top vendors by spend: {top_v_text}. "
        f"Overall procurement performance indicates opportunities in vendor optimization and currency risk management."
    )

# ---------- PDF class ----------
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

# ---------- Charts ----------
def generate_dashboard_charts(k, risk):
    charts = []
    try:
        if k.get("monthly"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(7,3))
                months = list(k["monthly"].keys())
                vals = list(k["monthly"].values())
                plt.plot(months, vals, marker="o")
                plt.title("Monthly Spend Trend")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout(); plt.savefig(tmp.name, bbox_inches="tight"); charts.append(tmp.name)
                plt.close()
    except Exception:
        plt.close()
    try:
        if k.get("top_v"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                top5 = dict(list(k["top_v"].items())[:5])
                plt.figure(figsize=(7,3))
                plt.bar(top5.keys(), top5.values())
                plt.title("Top 5 Vendors")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout(); plt.savefig(tmp.name, bbox_inches="tight"); charts.append(tmp.name)
                plt.close()
    except Exception:
        plt.close()
    try:
        if risk.get("breakdown"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(7,3))
                plt.bar(risk["breakdown"].keys(), risk["breakdown"].values())
                plt.title("Risk Breakdown")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout(); plt.savefig(tmp.name, bbox_inches="tight"); charts.append(tmp.name)
                plt.close()
    except Exception:
        plt.close()
    return charts

# ---------- PDF generator ----------
def generate_pdf(ai_text, insights_list, k, risk, charts, company):
    pdf = PDF(); pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica","B",20); pdf.cell(0,15,"Procurement Analytics Report",ln=True,align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica","",12); pdf.cell(0,10,f"Company: {company}",ln=True,align="C")
    pdf.cell(0,8,f"Generated On: {datetime.date.today():%d-%b-%Y}",ln=True,align="C")
    pdf.ln(6)
    # AI tag on cover
    pdf.set_font("Helvetica","I",9); pdf.cell(0,6,AI_TAGLINE,ln=True)
    pdf.ln(6)
    # Procurement Insights Summary (concise)
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Procurement Insights Summary",ln=True)
    pdf.set_font("Helvetica","",12)
    for ins in insights_list:
        pdf.multi_cell(0,7,ins)
    pdf.ln(4)
    # Executive Summary
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Executive Summary",ln=True)
    pdf.set_font("Helvetica","",12); pdf.multi_cell(0,7,ai_text); pdf.ln(4)
    # Key Performance Metrics
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Key Performance Metrics",ln=True)
    pdf.set_font("Helvetica","",12)
    metrics = [
        f"Total Spend: {k['total_spend']:,.2f} {k['dominant']}",
        f"Records: {k['records']}",
        f"Vendors (Top listed): {len(k['top_v'])}",
        f"Materials (Top listed): {len(k['top_m'])}",
        f"Risk Score: {risk['score']:.1f} ({risk['band']})"
    ]
    for m in metrics: pdf.cell(0,7,"- "+m,ln=True)
    pdf.ln(4)
    # Critical Findings
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Critical Findings",ln=True)
    pdf.set_font("Helvetica","",12)
    pdf.multi_cell(0,7,"Top vendor concentration and currency exposure require review; consider procurement optimization strategies.")
    pdf.ln(4)
    # Top Performing Vendors
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Top Performing Vendors",ln=True)
    pdf.set_font("Helvetica","",12)
    for v, val in k["top_v"].items(): pdf.cell(0,7,f"- {v}: {val:,.2f} {k['dominant']}",ln=True)
    pdf.ln(4)
    # Efficiency Analysis (natural sentences)
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Efficiency Analysis",ln=True)
    pdf.set_font("Helvetica","",12)
    eff = compute_efficiency_summary(k["df"])
    if eff:
        # most efficient = min avg cost; least efficient = max avg cost
        sorted_eff = sorted(eff.items(), key=lambda x: x[1]["avg_cost"])
        most_v, most_d = sorted_eff[0]
        least_v, least_d = sorted_eff[-1]
        # savings estimate if least matched most
        units = least_d["units_est"]
        gap = least_d["avg_cost"] - most_d["avg_cost"]
        est_savings = units * gap if units>0 and gap>0 else 0.0
        pdf.multi_cell(0,7,
            f"Most Efficient: {most_v} leads with {most_d['avg_cost']:.2f} cost-per-unit despite {most_d['total_spend']:,.0f} total spend, representing the efficiency benchmark."
        )
        pdf.multi_cell(0,7,
            f"Least Efficient: {least_v} shows {least_d['avg_cost']:.2f} cost-per-unit, presenting optimization opportunity with estimated annual saving of {est_savings:,.0f} (vendor currency)."
        )
    else:
        pdf.multi_cell(0,7,"Efficiency data not available.")
    pdf.ln(4)
    # Material Category Performance
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Material Category Performance",ln=True)
    pdf.set_font("Helvetica","",12)
    mat = compute_material_performance(k["df"])
    if mat:
        for m, val in list(mat.items())[:10]:
            pdf.cell(0,7,f"- {m}: {val:,.2f} {k['dominant']}",ln=True)
    else:
        pdf.multi_cell(0,7,"No material performance data available.")
    pdf.ln(4)
    # Dashboard Charts (fit)
    if charts:
        pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Dashboard Charts",ln=True)
        for ch in charts:
            try:
                y = pdf.get_y()
                pdf.image(ch, x=20, y=y+4, w=170)
                pdf.ln(78)
            except Exception:
                continue
    pdf.ln(4)
    # Risk Summary
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Risk Summary",ln=True)
    pdf.set_font("Helvetica","",12)
    pdf.cell(0,7,f"Risk Score: {risk['score']:.1f} ({risk['band']})",ln=True)
    for k_, v_ in risk.get("breakdown", {}).items(): pdf.cell(0,7,f"- {k_}: {v_:.1f}",ln=True)
    # AI tag at end
    pdf.ln(6); pdf.set_font("Helvetica","I",9); pdf.cell(0,6,AI_TAGLINE,ln=True)
    out = io.BytesIO(pdf.output(dest="S").encode("latin-1","ignore")); out.seek(0)
    return out

# ---------- APP FLOW ----------
st.subheader("🔐 Verify Access Key")
col1, col2 = st.columns([3, 1])
with col1:
    key = st.text_input("Enter access key", type="password")
with col2:
    if st.button("Verify"):
        if key and key.strip() in VALID_KEYS:
            st.session_state["verified"] = True
            st.success("Access verified.")
            st.rerun()
        else:
            st.error("Invalid key.")

if not st.session_state["verified"]:
    st.stop()

file = st.file_uploader("Upload Procurement File (CSV/XLSX)", type=["csv","xlsx"])
if not file:
    st.info("Please upload your procurement extract.")
    st.stop()

company = st.text_input("Enter Company Name", "ABC Manufacturing Pvt Ltd")

try:
    # read file
    if file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    k = compute_kpis(df)
    risk = compute_risk(k)
    ai_text = generate_ai_text(k)

    # compute insights
    insights = []
    insights.append(spend_trend_insight(k.get("monthly", {})))
    insights.append(vendor_dependency_insight(k.get("top_v", {}), k.get("total_spend", 0.0)))
    insights.append(currency_exposure_insight(k.get("totals", {})))
    eff_summary = compute_efficiency_summary(k["df"])
    insights.append(efficiency_insight(eff_summary))
    insights.append(material_mix_insight(k.get("top_m", {}), k.get("total_spend", 0.0)))
    insights.append(current_month_snapshot(k.get("monthly", {}), k.get("top_v", {})))

    # on-screen: Procurement Insights Summary (concise)
    st.markdown("## Procurement Insights Summary")
    for ins in insights:
        st.write("- " + ins)
    st.caption(AI_TAGLINE)

    # Executive summary and key metrics
    st.markdown("## Executive Summary")
    st.write(ai_text)

    st.markdown("### Key Performance Metrics")
    cols = st.columns(5)
    metrics_vals = [
        f"{k['total_spend']:,.2f} {k['dominant']}",
        k["records"],
        len(k["top_v"]),
        len(k["top_m"]),
        f"{risk['score']:.1f} ({risk['band']})"
    ]
    labels = ["Total Spend", "Records", "Vendors", "Materials", "Risk Score"]
    for c, label, val in zip(cols, labels, metrics_vals):
        c.metric(label, str(val))

    st.markdown("### Critical Findings")
    st.write("High vendor concentration and currency exposure detected; recommend supplier diversification and currency hedging considerations.")
    st.markdown("### Efficiency Analysis")
    if eff_summary:
        sorted_eff = sorted(eff_summary.items(), key=lambda x: x[1]["avg_cost"])
        most_v, most_d = sorted_eff[0]
        least_v, least_d = sorted_eff[-1]
        units = least_d["units_est"]
        gap = least_d["avg_cost"] - most_d["avg_cost"]
        est_savings = units * gap if units > 0 and gap > 0 else 0.0
        st.write(f"**Most Efficient:** {most_v} — {most_d['avg_cost']:.2f} cost/unit; total spend {most_d['total_spend']:,.0f}.")
        st.write(f"**Least Efficient:** {least_v} — {least_d['avg_cost']:.2f} cost/unit; estimated annual saving ≈ {est_savings:,.0f} (vendor currency).")
    else:
        st.write("Efficiency data not available.")

    st.markdown("### Material Category Performance")
    mat_perf = compute_material_performance(k["df"])
    if mat_perf:
        st.dataframe(pd.DataFrame.from_dict(mat_perf, orient="index", columns=["Spend"]).sort_values("Spend", ascending=False).head(10))
    else:
        st.write("No material performance data available.")

    st.markdown("### Dashboard Charts")
    charts = generate_dashboard_charts(k, risk)
    if charts:
        captions = []
        if k.get("monthly"): captions.append("Spend Trend")
        if k.get("top_v"): captions.append("Top Vendors")
        if risk.get("breakdown"): captions.append("Risk Breakdown")
        st.image(charts, caption=captions, use_container_width=True)
    else:
        st.info("Not enough data to generate charts.")

    st.caption(AI_TAGLINE)

    # PDF generation
    if st.button("📄 Generate PDF Report"):
        pdf_buf = generate_pdf(ai_text, insights, k, risk, charts, company)
        safe_name = company.strip().replace(" ", "_") or "Company"
        st.download_button("Download Report", pdf_buf, file_name=f"{safe_name}_Procurement_Report.pdf", mime="application/pdf")

except Exception:
    st.error("Error generating report:")
    st.text(traceback.format_exc())

