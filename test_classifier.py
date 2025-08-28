import pandas as pd
import numpy as np
from training.classifier import random_forest_classifier, evaluate_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_random_forest_classifier_roc_auc():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    y = pd.Series(np.where(y == 1, 1, 0)) 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clusters_train = np.random.randint(0, 2, size=len(X_train))
    clusters_test = np.random.randint(0, 2, size=len(X_test))

    model1, model2, uri1, uri2 = random_forest_classifier(
        X_train[clusters_train == 0],
        y_train[clusters_train == 0],
        X_train[clusters_train == 1],
        y_train[clusters_train == 1],
    )

    metrics = evaluate_model(
    model1,
    model2,
    X_test[clusters_test == 0],
    y_test[clusters_test == 0],
    X_test[clusters_test == 1],
    y_test[clusters_test == 1]
    )

    assert metrics["cluster_0"]["roc_auc"] > 0.5, f"Cluster 0 ROC AUC too low: {metrics['cluster_0']['roc_auc']}"
    assert metrics["cluster_1"]["roc_auc"] > 0.5, f"Cluster 1 ROC AUC too low: {metrics['cluster_1']['roc_auc']}"