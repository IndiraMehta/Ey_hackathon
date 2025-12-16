from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "ey_hackathon_db"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

raw_hospitals_gov = db["raw_hospitals_gov"]
raw_kaggle_medical = db["raw_kaggle_medical"]
raw_phc = db["raw_phc"]
