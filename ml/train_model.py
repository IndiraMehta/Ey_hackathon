import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from feature_engineering import build_features

# -----------------------------
# Utility: find column safely
# -----------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -----------------------------
# Load datasets
# -----------------------------
gov = pd.read_csv("../data/raw/hospitals_gov/1_gov_data.csv", low_memory=False)
kaggle = pd.read_csv("../data/raw/kaggle_medical/2_kaggle_data.csv", low_memory=False)

# Normalize column names
gov.columns = gov.columns.str.lower().str.strip()
kaggle.columns = kaggle.columns.str.lower().str.strip()

# Auto-detect columns
gov_name = find_col(gov, ["hospital_name", "name", "facility_name"])
gov_addr = find_col(gov, ["address", "full_address"])
gov_pin  = find_col(gov, ["pincode", "pin", "postal_code", "zip"])

kag_name = find_col(kaggle, ["hospital_name", "name", "facility_name"])
kag_addr = find_col(kaggle, ["address", "full_address"])
kag_pin  = find_col(kaggle, ["pincode", "pin", "postal_code", "zip"])

print("Detected columns:")
print("GOV:", gov_name, gov_addr, gov_pin)
print("KAG:", kag_name, kag_addr, kag_pin)

# Clean columns
for df, cols in [
    (gov, [gov_name, gov_addr, gov_pin]),
    (kaggle, [kag_name, kag_addr, kag_pin])
]:
    for c in cols:
        if c:
            df[c] = df[c].astype(str).fillna("")

# -----------------------------
# Build training data
# -----------------------------
X, y = [], []

# POSITIVE samples (same record vs itself)
POS = 300
for _ in range(POS):
    g = gov.sample(1).iloc[0]
    X.append(
        build_features(
            {
                "name": g[gov_name] if gov_name else "",
                "address": g[gov_addr] if gov_addr else "",
                "pincode": g[gov_pin] if gov_pin else "",
                "lat": None,
                "lng": None
            },
            {
                "name": g[gov_name] if gov_name else "",
                "address": g[gov_addr] if gov_addr else "",
                "pincode": g[gov_pin] if gov_pin else "",
                "lat": None,
                "lng": None
            }
        )
    )
    y.append(1)

# NEGATIVE samples (random mismatch)
NEG = 300
for _ in range(NEG):
    g = gov.sample(1).iloc[0]
    k = kaggle.sample(1).iloc[0]

    X.append(
        build_features(
            {
                "name": g[gov_name] if gov_name else "",
                "address": g[gov_addr] if gov_addr else "",
                "pincode": g[gov_pin] if gov_pin else "",
                "lat": None,
                "lng": None
            },
            {
                "name": k[kag_name] if kag_name else "",
                "address": k[kag_addr] if kag_addr else "",
                "pincode": k[kag_pin] if kag_pin else "",
                "lat": None,
                "lng": None
            }
        )
    )
    y.append(0)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "model.joblib")

print("✅ Model trained and saved as model.joblib")
