
import streamlit as st
import joblib
from ml.feature_engineering import build_features
from ml.trust_scoring import compute_trust

model = joblib.load("ml/model.joblib")

st.set_page_config(page_title="Healthcare Provider Validation", layout="wide")
st.title("🏥 Healthcare Provider Validation System")

# -------- Facility Input --------
st.subheader("1️⃣ Facility Details")
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Hospital Name *")
    address = st.text_area("Address *")
    city = st.text_input("City")
    state = st.text_input("State")

with col2:
    pincode = st.text_input("Pincode *")
    lat = st.number_input("Latitude", value=0.0, format="%.6f")
    lng = st.number_input("Longitude", value=0.0, format="%.6f")

# -------- Metadata --------
st.subheader("2️⃣ Source Metadata")
source = st.selectbox("Data Source", ["gov", "kaggle", "manual"])
last_updated = st.date_input("Last Updated")

# -------- Contact --------
st.subheader("3️⃣ Contact (Optional)")
phone = st.text_input("Phone")
email = st.text_input("Email")

# -------- Run Validation --------
if st.button("🔍 Validate Record"):
    record_1 = {
        "name": name,
        "address": address,
        "pincode": pincode,
        "lat": lat,
        "lng": lng
    }

    # self-match for demo
    record_2 = record_1.copy()

    features = build_features(record_1, record_2)
    dup_prob = model.predict_proba([features])[0][1]

    trust = compute_trust(
        {
            "name": [(name, source)],
            "address": [(address, source)],
            "pincode": [(pincode, source)]
        },
        ml_confidence=dup_prob
    )

    st.markdown("---")
    st.subheader("📊 Results")

    st.metric("Duplicate Confidence", f"{dup_prob:.2f}")
    st.metric("Overall Trust Score", f"{trust['overall_trust']:.2f}")

    st.json(trust)

    if trust["overall_trust"] >= 0.85:
        st.success("✅ AUTO-APPROVED (Golden Record)")
    elif trust["overall_trust"] >= 0.65:
        st.warning("🟡 NEEDS MANUAL REVIEW")
    else:
        st.error("🔴 EXCEPTION QUEUE")
