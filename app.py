# MAIN UI

st.title("📊 Executive Procurement Dashboard")
company_name = st.text_input("Enter Company Name:", "ABC Manufacturing Pvt Ltd")
f = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
if not f:
    st.stop()

df = pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
k = compute_kpis(df)
risk = compute_procurement_risk(df, k)
gauge = plot_risk_gauge(risk["score"])
charts = [gauge]

# --- Pie Chart with Safe Check ---
if k["totals"] and sum(k["totals"].values()) > 0:
    fig, ax = plt.subplots()
    ax.pie(
        k["totals"].values(),
        labels=k["totals"].keys(),
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Currency Distribution")
    fig.savefig("chart_currency.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    charts.append("chart_currency.png")
else:
    st.warning("No currency data available for pie chart.")

# --- Top Vendors Bar Chart ---
if k["top_v"]:
    fig, ax = plt.subplots()
    ax.barh(
        list(k["top_v"].keys())[::-1],
        list(k["top_v"].values())[::-1],
        color="#2E7D32"
    )
    ax.set_title("Top Vendors by Spend")
    fig.savefig("chart_vendors.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    charts.append("chart_vendors.png")

# --- Top Materials Bar Chart ---
if k["top_m"]:
    fig, ax = plt.subplots()
    ax.bar(
        list(k["top_m"].keys()),
        list(k["top_m"].values()),
        color="#1565C0"
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Top Materials by Quantity/Spend")
    fig.savefig("chart_materials.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    charts.append("chart_materials.png")

# --- Monthly Trend Line Chart ---
if k["monthly"]:
    fig, ax = plt.subplots()
    ax.plot(
        list(k["monthly"].keys()),
        list(k["monthly"].values()),
        marker="o"
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Monthly Purchase Trend")
    fig.savefig("chart_monthly.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    charts.append("chart_monthly.png")

# --- KPI Cards ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", k["records"])
c2.metric("Spend", f"{k['total_spend']:,.2f} {k['dominant']}")
c3.metric("Top Vendor", next(iter(k["top_v"]), "N/A"))
c4.metric("Risk", f"{risk['score']:.0f} ({risk['band']})")

st.subheader("Procurement Risk Index Gauge")
st.image(gauge, use_container_width=True, caption="Procurement Risk Index Gauge")

st.subheader("Procurement Risk Breakdown")
st.table(
    pd.DataFrame.from_dict(risk["breakdown"], orient="index", columns=["Score"])
    .reset_index().rename(columns={"index": "Metric"})
)

st.subheader("Visual Highlights")
for ch in charts[1:]:
    st.image(ch, use_container_width=True)

st.subheader("AI Insights")
ai = generate_ai(k)
st.markdown(ai.replace("\n", "  \n"))
summary = ai[:1000]

pdf = generate_pdf(ai, k, charts, company_name, summary, risk)
st.download_button("📄 Download Full Executive Report", pdf, "SAP_Automatz_Executive_Report.pdf", "application/pdf")
