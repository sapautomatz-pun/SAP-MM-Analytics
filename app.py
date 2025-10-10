# ==============================================
# SAP Automatz – Procurement Analytics (v3.3)
# ==============================================
#  • Integrates with Razorpay backend for access validation
#  • Supports auto-prefilled ?access=<key> in URL
#  • Displays expiry and plan info
#  • Loads dashboard only if access verified
# ==============================================

import streamlit as st
import requests
import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BACKEND_VERIFY_URL = "https://sapautomatz-backend.onrender.com/verify_access"

st.set_page_config(page_title="SAP MM Procurement Analytics",
                   page_icon="📊", layout="wide")

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def check_access(key: str):
    """Query backend to verify access key."""
    try:
        res = requests.get(BACKEND_VERIFY_URL, params={"key": key}, timeout=10)
        return res.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def show_access_form():
    """Display login form for access key entry."""
    st.markdown("### 🔐 Enter your Access Key")
    query_params = st.experimental_get_query_params()
    default_key = query_params.get("access", [""])[0]
    access_key = st.text_input("Access Key", value=default_key, placeholder="e.g., SAPMM-20251010120000")

    if st.button("Verify Access"):
        with st.spinner("Verifying your subscription..."):
            result = check_access(access_key)

        if result.get("status") == "ok":
            st.success(f"✅ Access verified! Welcome {result.get('name')} ({result.get('plan')})")
            st.info(f"Valid until: {result.get('expiry')}")
            st.session_state["verified"] = True
            st.session_state["user_info"] = result
            st.rerun()

        elif result.get("status") == "expired":
            st.error(f"❌ Your plan expired on {result.get('expiry')}. Please renew your access.")
        elif result.get("status") == "invalid":
            st.error("❌ Invalid Access Key. Please check your email.")
        else:
            st.error(f"⚠️ Error verifying access: {result.get('message', 'Unknown error')}")
    else:
        st.caption("You'll receive your access key by email after payment.")

# ---------------------------------------------------------------------
# Main Dashboard Placeholder
# ---------------------------------------------------------------------
def show_dashboard(user):
    st.title("📊 SAP MM Procurement Analytics")
    st.markdown(f"**Plan:** {user.get('plan').capitalize()} | **Valid till:** {user.get('expiry')}")
    st.divider()

    st.subheader("🔎 Upload your SAP Procurement Data")
    uploaded_file = st.file_uploader("Upload Purchase Order / GRN file (.csv or .xlsx)", type=["csv", "xlsx"])

    if uploaded_file:
        st.success("✅ File uploaded successfully.")
        # Placeholder for your analytics logic
        st.markdown("### Procurement KPIs")
        st.write("⚙️ Generating insights... (demo placeholder)")
        # Example of where to insert AI analytics
        # df = pd.read_csv(uploaded_file) or pd.read_excel(uploaded_file)
        # display_kpis(df)
        # plot_charts(df)
    else:
        st.info("Please upload your SAP PO / GRN dataset to view analytics.")

    st.divider()
    st.caption("© 2025 SAP Automatz | Powered by SAP GEN AI")

# ---------------------------------------------------------------------
# App Logic
# ---------------------------------------------------------------------
st.sidebar.image("https://sapautomatz.github.io/SAP-MM-Reconciliation-tool/logo.png",
                 use_column_width=True)
st.sidebar.markdown("### SAP Automatz\nAI-Driven Procurement Intelligence")

if "verified" not in st.session_state or not st.session_state["verified"]:
    show_access_form()
else:
    user_info = st.session_state.get("user_info", {})
    if user_info:
        show_dashboard(user_info)
    else:
        st.session_state.clear()
        st.rerun()
