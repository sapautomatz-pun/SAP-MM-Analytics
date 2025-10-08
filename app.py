import streamlit as st
import pandas as pd
from io import BytesIO
import base64
import datetime

st.set_page_config(page_title="SAP MM Procurement Analytics", layout="wide")

st.title("📊 SAP MM Procurement Analytics - Auto Report Generator")
st.markdown("Upload your **Purchase Order or GRN** file (.csv or .xlsx) to automatically generate a Procurement Performance Summary.")

# ---- Utility Functions ----

def generate_summary(df):
    """Perform simple procurement analytics"""
    summary = {}
    df.columns = [c.strip().lower() for c in df.columns]

    # Sample analytics: PO value per vendor
    if 'vendor' in df.columns and 'po value' in df.columns:
        vendor_summary = df.groupby('vendor')['po value'].sum().reset_index()
        summary['total_po_value'] = df['po value'].sum()
        summary['top_vendors'] = vendor_summary.sort_values('po value', ascending=False).head(5)
    else:
        vendor_summary = pd.DataFrame()

    # PO count by month
    if 'posting date' in df.columns:
        df['posting date'] = pd.to_datetime(df['posting date'], errors='coerce')
        monthly = df.groupby(df['posting date'].dt.to_period('M')).size().reset_index(name='count')
        monthly['posting date'] = monthly['posting date'].astype(str)
    else:
        monthly = pd.DataFrame()

    return summary, vendor_summary, monthly


def generate_excel_report(vendor_summary, monthly_summary):
    """Generate Excel report"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        vendor_summary.to_excel(writer, index=False, sheet_name='Top Vendors')
        monthly_summary.to_excel(writer, index=False, sheet_name='Monthly Trend')
    output.seek(0)
    return output


def make_download_link(data, filename):
    """Convert report to downloadable link"""
    b64 = base64.b64encode(data.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download {filename}</a>'


# ---- Upload Section ----

uploaded_file = st.file_uploader("Upload SAP PO or GRN file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
    
    summary, vendor_summary, monthly = generate_summary(df)
    
    if not vendor_summary.empty:
        st.subheader("📈 Top 5 Vendors by PO Value")
        st.dataframe(vendor_summary)
        st.bar_chart(vendor_summary.set_index('vendor')['po value'])

    if not monthly.empty:
        st.subheader("📅 Monthly PO Count Trend")
        st.line_chart(monthly.set_index('posting date')['count'])

    excel_report = generate_excel_report(vendor_summary, monthly)
    download_link = make_download_link(excel_report, f"Procurement_Report_{datetime.date.today()}.xlsx")
    st.markdown(download_link, unsafe_allow_html=True)
else:
    st.info("Upload your SAP Purchase Order or GRN data to begin analysis.")

# ---- Optional: API-style endpoint simulation ----
st.markdown("---")
st.caption("This app can also be called via Make.com HTTP POST (using the same analytics logic).")
