# ==============================================
# SAP Automatz – Procurement Analytics (v3.8)
# ==============================================
# • Updated for Streamlit 2025 (no warnings)
# • Secure Access Verification via backend
# • Dynamic banner showing days left till expiry
# • Built-in Renew Access CTA
# ==============================================

import streamlit as st
import requests
import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BACKEND_VERIFY_URL = "https://sapautomatz-backend.onrender.com/verify_access"
RENEW_URL = "https://rzp.io/l/sapautomatz"   # Razorpay renewal link

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

def show_renew_button(label="🔁 Renew Access"):
    """Render a consistent Renew Access button."""
    st.markdown(
        f"""
        <a href="{RENEW_URL}" target="_blank" style="
            text-decoration:none;
            ">
            <button style="
                background-color:#007bff;
                color:white;
                padding:10px 24px;
                border:none;
                border-radius:6px;
                font-size:16px;
                font-weight:bold;
                cursor:pointer;">
                {label}
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Banner Renderer
# ---------------------------------------------------------------------
def show_banner(user):
    """Display a colored banner with expiry info."""
    expiry = user.get("expiry")
    plan = user.get("plan", "").capitalize()
    if expiry == "Lifetime":
        st.markdown(
            f"<div style='background-color:#28a745;padding:12px;border-radius:8px;text-align:center;color:white;font-weight:bold;'>"
            f"💎 Lifetime Access Activated — Enjoy Unlimited Use</div>",
            unsafe_allow_html=True
        )
        return

    try:
        expiry_date = datetime.date.fromisoformat(expiry)
        today = datetime.date.today()
        days_left = (expiry_date - today).days

        if days_left > 10:
            color = "#28a745"  # Green
        elif 4 <= days_left <= 10:
            color = "#ff9800"  # Orange
        else:
            color = "#dc3545"  # Red

        banner_text = f"⏳ Your {plan} plan expires in {days_left} day{'s' if days_left != 1 else ''} (on {expiry})."
        if days_left <= 3:
            banner_text += " Please renew soon to avoid interruption."

        st.markdown(
            f"""
            <div style='background-color:{color};padding:10px;border-radius:8px;text-align:center;color:white;font-weight:bold;'>
                {banner_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if days_left <= 5:
            show_renew_button("🔁 Renew Now")

    except Exception:
        pass

# ---------------------------------------------------------------------
# Access Form
# ---------------------------------------------------------------------
def show_access_form():
    """Display login form for access key entry."""
    st.markdown("### 🔐 Enter your Access Key")

    query_params = st.query_params
    default_key = (
        query_params.get("access", [""])[0]
        if isinstance(query_params.get("access"), list)
        else query_params.get("access", "")
    )

    access_key = st.text_input("Access Key", value=default_key, placeholder="e.g., SAPMM-20251010120000")

    if st.button("Verify Access", use_container_width=True):
        with st.spinner("Verifying your subscription..."):
            result = check_access(access_key)

        # --- Valid Access ---
        if result.get("status") == "ok":
            st.success(f"✅ Access verified! Welcome {result.get('name')} ({result.get('plan')})")
            st.info(f"Valid until: {result.get('expiry')}")
            st.session_state["verified"] = True
            st.session_state["user_info"] = result
            st.rerun()

        # --- Expired Access ---
        elif result.get("status") == "expired":
            st.error(f"❌ Your plan expired on {result.get('expiry')}.")
            show_renew_button()

        # --- Invalid Key ---
        elif result.get("status") == "invalid":
            st.error("❌ Invalid Access Key. Please check your email.")

        # --- Backend Error ---
        else:
            st.error(f"⚠️ Error verifying access: {result.get('message', 'Unknown error')}")
    else:
        st.caption("You'll receive your access key by email after payment.")

# ---------------------------------------------------------------------
# Dashboard Logic
# ---------------------------------------------------------------------
def show_dashboard(user):
    """Main dashboard after access verification."""
    show_banner(user)
    st.title("📊 SAP MM Procurement Analytics")
    st.markdown(f"**Plan:** {user.get('plan').capitalize()} | **Valid till:** {user.get('expiry')}")
    st.divider()

    st.subheader("🔎 Upload your SAP Procurement Data")
    uploaded_file = st.file_uploader(
        "Upload Purchase Order / GRN file (.csv or .xlsx)",
        type=["csv", "xlsx"],
        label_visibility="visible"
    )

    if uploaded_file:
        st.success("✅ File uploaded successfully.")
        st.markdown("### Procurement KPIs")
        st.write("⚙️ Generating insights... (demo placeholder)")
        # Example future integration:
        # df = pd.read_csv(uploaded_file)
        # display_kpis(df)
        # plot_charts(df)
    else:
        st.info("Please upload your SAP PO / GRN dataset to view analytics.")

    st.divider()
    st.caption("© 2025 SAP Automatz | Powered by SAP GEN AI")

# ---------------------------------------------------------------------
# Sidebar Branding
# ---------------------------------------------------------------------
st.sidebar.image(
    "https://sapautomatz.github.io/SAP-MM-Reconciliation-tool/logo.png",
    use_container_width=True
)
st.sidebar.markdown("### SAP Automatz\nAI-Driven Procurement Intelligence")

# ---------------------------------------------------------------------
# Main App Logic
# ---------------------------------------------------------------------
if "verified" not in st.session_state or not st.session_state["verified"]:
    show_access_form()
else:
    user_info = st.session_state.get("user_info", {})
    if user_info:
        show_dashboard(user_info)
    else:
        st.session_state.clear()
        st.rerun()
