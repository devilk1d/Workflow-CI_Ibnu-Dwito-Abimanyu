import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import argparse
import joblib

def train_advance(data_path):
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    X = df["clean_text"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_tfidf, y_train)

    best_model = grid_search.best_estimator_

    # =====================
    # MLflow Logging
    # =====================
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    test_acc = best_model.score(X_test_tfidf, y_test)
    mlflow.log_metric("test_accuracy", test_acc)

    os.makedirs("artifacts", exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test_tfidf, y_test, cmap="Blues"
    )
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Classification Report
    report = classification_report(y_test, best_model.predict(X_test_tfidf))
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Save vectorizer
    vec_path = "artifacts/tfidf_vectorizer.pkl"
    joblib.dump(tfidf, vec_path)
    mlflow.log_artifact(vec_path)

    # Log Model
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="spam_model_rf",
        registered_model_name="SpamEmailDetector",
    )

    print(f"Training finished. Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    train_advance(args.data_path)