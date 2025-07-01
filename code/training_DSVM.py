import timeit
from collections import Counter
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from DSVM import DSVM


def predict_with_dsvm_list(dsvm_list, X):
    n_samples = X.shape[0]
    votes = [[] for _ in range(n_samples)]

    for dsvm in dsvm_list:
        preds = dsvm.predict(X)
        for i, pred in enumerate(preds):
            votes[i].append(pred)

    majority_votes = []
    for vote in votes:
        vote_count = Counter(vote)
        majority_votes.append(vote_count.most_common(1)[0][0])
    return np.array(majority_votes)


def train_dsvm(dataset_path: str = 'data_in_use/tmp_file.csv', model_path: str = "best_dsvm.pkl"):
    df = pd.read_csv(dataset_path)
    features = df.drop(columns=['Cluster']).values.astype(float)
    labels = df['Cluster'].values.astype(int)
    classes = np.unique(labels)

    X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.2, random_state=42)

    svm_classifiers = []
    for c1, c2 in combinations(classes, 2):
        svm_classifiers.append(DSVM(c1, c2, kernel='rbf', C=1.0, gamma='scale'))

    for dsvm in svm_classifiers:
        dsvm.train(X_train, y_train)

    joblib.dump(svm_classifiers, model_path)
    print(f"Saved DSVM ensemble to {model_path}")

    final_predictions = predict_with_dsvm_list(svm_classifiers, X_eval)
    accuracy = accuracy_score(y_eval, final_predictions)
    print("Evaluation accuracy:", accuracy)


if __name__ == "__main__":
    s = timeit.default_timer()
    train_dsvm('data_in_use/tmp_file.csv', "best_dsvm.pkl")
    e = timeit.default_timer()
    print("ELAPSED:", e - s)
