import pandas as pd
from datetime import datetime
from database.mongo import raw_hospitals_gov

FILE_PATH = "data/raw/hospitals_gov/1_gov_data.csv"

df = pd.read_csv(FILE_PATH)

records = df.to_dict(orient="records")

for r in records:
    r["source"] = "government"
    r["reliability_score"] = 0.9
    r["last_updated"] = datetime.utcnow()

raw_hospitals_gov.insert_many(records)

print(f"Inserted {len(records)} government records")
