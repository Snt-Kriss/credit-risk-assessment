from fastapi import FastAPI, HTTPException
from serve.models.model import CustomerFeatures, PredictionResponse
from serve.mlflow.handler import MlflowHandler
import pandas as pd
import logging


log_format= "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger= logging.getLogger(__name__)

app= FastAPI(title= "Credit Risk Classifier API", version="1.0.0")

handler= MlflowHandler()

@app.get("/health", status_code=200)
async def healthcheck():
    try:
        status= handler.check_mlflow_health()
        return {"serviceStatus": "OK", "mlflow": status}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="MLflow health check failed")
    

@app.post("/cluster", status_code=200)
def get_user_cluster(features: CustomerFeatures):
    """
    Debug endpoint: only return cluster assignment.
    """
    try:
        user_features = [
            features.age,
            features.job,
            features.credit_amount,
            features.duration
        ]
        cluster_id = handler.assign_cluster(user_features)
        return {"cluster": int(cluster_id)}
    except Exception as e:
        logger.error(f"Cluster assignment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cluster assignment failed: {str(e)}")
    

@app.post("/classify", response_model=PredictionResponse, status_code=200)
async def classify_customer(features: CustomerFeatures):
    try:
        user_features= [
            features.age,
            features.job,
            features.credit_amount,
            features.duration
        ]

        cluster_id= handler.assign_cluster(user_features)
        logger.info(f"User assigned to cluster {cluster_id}")

        model= handler.get_model_for_cluster(cluster_id)
        pred= model.predict([user_features])[0]

        prob= 0.0
        try:
            prob= model.predict_proba([user_features])[0][1]
        except Exception:
            logger.warning("Model does not support predict_proba")

        return PredictionResponse(
            risk="Good" if pred==1 else "Bad",
            probability= round(float(prob), 4)
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")