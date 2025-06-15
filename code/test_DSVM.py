import multiprocessing

import joblib
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from sklearn.metrics import accuracy_score

from DSVM import DSVM  # Ensure DSVM.py is in your PYTHONPATH or working directory


def predict_with_dsvm_list(dsvm_list, X):
    n_samples = X.shape[0]
    votes = [[] for _ in range(n_samples)]
    for dsvm in dsvm_list:
        preds = dsvm.predict(X)
        for i, pred in enumerate(preds):
            votes[i].append(pred)
    majority_votes = []
    for vote in votes:
        majority_votes.append(Counter(vote).most_common(1)[0][0])
    return np.array(majority_votes)


def test_dsvm(normalized_data_path: str = "data_in_use/data_12343658_1_022.csv", dsvm_model_path: str = "best_dsvm.pkl",
              save_csv: bool = True):
    multiprocessing.freeze_support()

    dsvm_list = joblib.load(dsvm_model_path)
    print("Loaded DSVM ensemble from", dsvm_model_path)
    df = pd.read_csv(normalized_data_path)
    features = df.drop(columns=['Cluster']).values.astype(float)
    labels = df['Cluster'].values.astype(int)

    predictions = predict_with_dsvm_list(dsvm_list, features)

    accuracy = accuracy_score(labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    df["Prediction"] = predictions

    if save_csv:
        parts = normalized_data_path.split("/")
        file = parts[-1]
        file_name, ext = file.split(".")
        new_file = file_name + "_dsvm." + ext
        parts[-1] = new_file
        out_path = "/".join(parts)
        print(f"Saving predictions to {out_path}")
        df.to_csv(out_path, index=False)


if __name__ == '__main__':
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_012.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_022.csv", save_csv=True)
