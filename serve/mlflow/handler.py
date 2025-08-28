from pprint import pprint
import mlflow
from mlflow.client import MlflowClient
from mlflow.pyfunc import PyFuncModel
import joblib


class MlflowHandler:
    def __init__(self, tracking_uri: str= "sqlite:///mlflow.db")-> None:
        self.tracking_uri= tracking_uri
        self.client= MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        
        self.cluster_model= joblib.load("cluster_model.joblib")

        self.cluster_to_model={
            0: "cluster_0_model",
            1: "cluster_1_model"
        }


    def check_mlflow_health(self)-> str:
        try:
            experiments= self.client.search_experiments()
            for rm in experiments:
                pprint(dict(rm), indent=4)
            return "Service is running and returning experiments"
        except Exception as e:
            return f"Error calling mlflow: {e}"
        
    def assign_cluster(self, user_features)-> int:
        """Assign user to cluster using KMeans model"""
        return self.cluster_model.predict([user_features])[0]
    
    def get_model_for_cluster(self, cluster_id: int)-> PyFuncModel:
        model_name= self.cluster_to_model.get(cluster_id)
        if not model_name:
            raise ValueError(f"No model was found for cluster {cluster_id}")
        try:
            return mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        except Exception as e:
            raise RuntimeError(f"Could not load model: {model_name}: {e}")
        
    def predict(self, user_features):
        cluster_id= self.assign_cluster(user_features)
        model= self.get_model_for_cluster(cluster_id)
        return model.predict([user_features])