# server/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# load the pipeline (make sure path is correct relative to app.py)
model = joblib.load("models/elasticnet_best.joblib")

class Features(BaseModel):
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Tutoring_Sessions: float
    Family_Income_Num: int
    Distance_from_Home_Num: int
    Motivation_Level_Num: int
    Parental_Involvement_Num: int
    Access_to_Resources_Num: int
    Physical_Activity_Num: int
    Peer_Influence: str
    Internet_Access: str
    Learning_Disabilities: str
    Gender: str

app = FastAPI()

@app.post("/predict")
def predict(feat: Features):
    df = pd.DataFrame([feat.dict()])
    yhat = model.predict(df)[0]
    return {"predicted_score": float(yhat)}
