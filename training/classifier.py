import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow

def load_data(filepath: str)-> pd.DataFrame:
    df= pd.read_csv(filepath)
    return df


def scale_numerical_data(df: pd.DataFrame)-> pd.DataFrame:
    numerical_data= df.select_dtypes(exclude='O')
    scaler= StandardScaler()
    scaled_data= scaler.fit_transform(numerical_data)
    return scaled_data

def cluster_analysis(credit: pd.DataFrame, df: pd.DataFrame, k: int=2)-> pd.DataFrame:
    kmeans= KMeans(n_clusters=k)
    cluster= kmeans.fit_predict(df)
    clusters, counts= np.unique(kmeans.labels_, return_counts=True)
    cluster_dict={}
    for i in range(len(clusters)):
        cluster_dict[i]= df[kmeans.labels_==i]

    credit['clusters']= pd.DataFrame(kmeans.labels_)
    df_scaled= pd.DataFrame(df)
    df_scaled['clusters']= credit['clusters']
    df_scaled['Risk']= credit['Risk']
    df_scaled.columns= ['Age', 'Job', 'CreditAmount', 'Duration', 'Clusters', 'Risk']

    joblib.dump(kmeans, "cluster_model.joblib")

    return df_scaled


def train_test_split_data(df_scaled: pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_scaled['Risk']= df_scaled['Risk'].replace({'good':1, 'bad':0})
    X= df_scaled.drop('Risk', axis=1)
    y= df_scaled.loc[:, ['Risk', 'Clusters']]

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def group_clusters(X_train: pd.DataFrame)-> pd.DataFrame:
    first_cluster_train= X_train[X_train.Clusters==0].iloc[:, :-1]
    second_cluster_train= X_train[X_train.Clusters==1].iloc[:, :-1]

    return first_cluster_train, second_cluster_train


def resample_first_cluster_data(first_cluster_train: pd.DataFrame,
                  y_train: pd.DataFrame, X_test: pd.DataFrame, 
                  y_test: pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train1= first_cluster_train
    y_train1= y_train[y_train.Clusters==0]['Risk']
    smote= SMOTEENN(random_state=42)
    X_train1, y_train1= smote.fit_resample(X_train1, y_train1)

    first_cluster_test= X_test[X_test.Clusters==0].iloc[:, :-1]
    

    X_test1= first_cluster_test
    y_test1= y_test[y_test.Clusters==0]['Risk']

    return X_train1, X_test1, y_train1, y_test1


def second_cluster_data(second_cluster_train: pd.DataFrame, 
              y_train: pd.DataFrame, y_test: pd.DataFrame,X_test: pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    second_cluster_test= X_test[X_test.Clusters==1].iloc[:, :-1]

    X_train2= second_cluster_train
    y_train2= y_train[y_train.Clusters==1]['Risk']

    X_test2= second_cluster_test
    y_test2= y_test[y_test.Clusters==1]['Risk']

    return X_train2, X_test2, y_train2, y_test2

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("credit-risk-classifier")

def random_forest_classifier(X_train1: pd.DataFrame, y_train1: pd.DataFrame,
                             X_train2: pd.DataFrame, y_train2: pd.DataFrame):
    model_cluster1= RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        max_features='log2',
        max_depth=5,
        criterion='entropy'
    )

    model_cluster2= RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        max_features='log2',
        max_depth=5,
        criterion='gini'
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="cluster_0_model") as run1:
        model_cluster1.fit(X_train1, y_train1)
        model1_uri= mlflow.get_artifact_uri("model")

    with mlflow.start_run(run_name="cluster_1_model") as run2:
        model_cluster2.fit(X_train2, y_train2)
        model2_uri= mlflow.get_artifact_uri("model")


    return model_cluster1, model_cluster2, model1_uri, model2_uri



def evaluate_model(model1, model2, X_test1, y_test1, X_test2, y_test2):
    # Cluster 0
    y_pred1 = model1.predict(X_test1)
    y_pred_proba1 = model1.predict_proba(X_test1)[:, 1]
    acc1 = accuracy_score(y_test1, y_pred1)
    auc1 = roc_auc_score(y_test1, y_pred_proba1)
    recall1= recall_score(y_test1, y_pred1)

    # Cluster 1
    y_pred2 = model2.predict(X_test2)
    y_pred_proba2 = model2.predict_proba(X_test2)[:, 1]
    acc2 = accuracy_score(y_test2, y_pred2)
    auc2 = roc_auc_score(y_test2, y_pred_proba2)
    recall2= recall_score(y_test2, y_pred2)

    with mlflow.start_run(run_name="evaluation", nested=True):
        mlflow.log_metric("accuracy_cluster_0", acc1)
        mlflow.log_metric("accuracy_cluster_1", acc2)
        mlflow.log_metric("recall_cluster_0", recall1)
        mlflow.log_metric("recall_cluster_1", recall2)

    return {
        "cluster_0": {"accuracy": acc1, "roc_auc": auc1},
        "cluster_1": {"accuracy": acc2, "roc_auc": auc2},
    }

def register_cluster_model(model, cluster_id):
    model_name= f"cluster_{cluster_id}_model"
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

    print(f"Registered {model_name} in MLflow")


def save_models(model1, model2, path: str= "model_cluster1.joblib", path2: str= "model_cluster2.joblib")-> None:
    joblib.dump(model1, path)
    joblib.dump(model2, path2)
    print(f"Models saved to {path} and {path2}")


if __name__=="__main__":
    df= load_data('./data/german_credit_data.csv')

    scaled_data= scale_numerical_data(df)

    df_scaled= cluster_analysis(df, scaled_data)
    X_train, X_test, y_train, y_test= train_test_split_data(df_scaled)

    first_cluster_train, second_cluster_train= group_clusters(X_train)

    X_train1, X_test1, y_train1, y_test1= resample_first_cluster_data(first_cluster_train, y_train, X_test, y_test)

    X_train2, X_test2, y_train2, y_test2= second_cluster_data(second_cluster_train, y_train, y_test, X_test)

    model1, model2, uri1, uri2= random_forest_classifier(X_train1, y_train1, X_train2, y_train2)

    evaluate_model(model1, model2, X_test1, y_test1, X_test2, y_test2)

    save_models(model1, model2)

    register_cluster_model(model1, cluster_id=0)
    register_cluster_model(model2, cluster_id=1)




