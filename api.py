from fastapi import FastAPI
import joblib
from ml.feature_engineering import build_features
from ml.trust_scoring import compute_trust
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
model = joblib.load("ml/model.joblib")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/validate")
def validate(data: dict):
    record = {
        "name": data["name"],
        "address": data["address"],
        "pincode": data["pincode"]
    }

    features = build_features(record, record)
    dup_prob = model.predict_proba([features])[0][1]

    trust = compute_trust(
        {
            "name": [(data["name"], data["source"])],
            "address": [(data["address"], data["source"])],
            "pincode": [(data["pincode"], data["source"])]
        },
        ml_confidence=dup_prob
    )

    return {
        "duplicate_probability": round(float(dup_prob), 3),
        "trust": trust
    }
