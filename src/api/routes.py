from fastapi import APIRouter,HTTPException
from src.api.schema import CustomerInput
from Prediciton_Pipeline import PredictionPipeline
from config import configcall
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
# CONFIG_PATH = BASE_DIR/"config.yaml"

# with open(CONFIG_PATH,"r") as file:
#     config = yaml.safe_load(file)

model_path = BASE_DIR/"tunemodels"/"xgboost.pkl"
preprocessor_path = BASE_DIR/"preprocessor.pkl"

router = APIRouter()

pp = PredictionPipeline(model_path=model_path,preprocessor_path=preprocessor_path)

@router.get("/health")
def health_check():
    """check whether the api is running"""

    return {"status":"healthy","message":  "Customer chrun prediction api is running"}

@router.post("/predict")
def predict_customer(customer:CustomerInput):
    """predict whether customer churn or not"""
    try:
        input_data = customer.model_dump()
        prediction = pp.predict(input_data)
        return prediction

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"prediction failed {e}")
