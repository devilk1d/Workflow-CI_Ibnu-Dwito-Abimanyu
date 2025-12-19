import os
import urllib.parse
import json
import mlflow
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# 1. KONFIGURASI MLFLOW (LOCAL)
# ============================================================

print("Initializing MLflow Tracking (Local)...")

# ============================================================
# 2. LOAD DATA
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='spam_email_dataset_cleaned.csv', help='Path to cleaned dataset CSV')
args = parser.parse_args()

df = pd.read_csv(args.data_path)

df["clean_text"] = df["clean_text"].fillna("").astype(str)
X = TfidfVectorizer().fit_transform(df["clean_text"])
y = df["label_str"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================================
# 3. HYPERPARAMETER GRID
# ============================================================

param_grid_nb = {"alpha": [0.1, 0.5, 1.0]}
param_grid_svm = {"C": [0.5, 1.0, 2.0]}

with open("param_grid_nb.json", "w") as f:
    json.dump(param_grid_nb, f, indent=2)

with open("param_grid_svm.json", "w") as f:
    json.dump(param_grid_svm, f, indent=2)

# ============================================================
# 4. PARENT RUN (HYPERPARAMETER TUNING)
# ============================================================

with mlflow.start_run(run_name="Hyperparameter_Tuning"):

    # --- Artefak global (ADVANCE) ---
    mlflow.log_artifact("param_grid_nb.json")
    mlflow.log_artifact("param_grid_svm.json")

    # ========================================================
    # 5. NAIVE BAYES TUNING (NESTED RUN)
    # ========================================================

    for alpha in param_grid_nb["alpha"]:
        with mlflow.start_run(
            run_name=f"NB_alpha_{alpha}",
            nested=True
        ):
            model = MultinomialNB(alpha=alpha)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            # ---- Manual Logging ----
            mlflow.log_param("model", "MultinomialNB")
            mlflow.log_param("alpha", alpha)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_spam", report["spam"]["precision"])
            mlflow.log_metric("recall_spam", report["spam"]["recall"])
            mlflow.log_metric("f1_spam", report["spam"]["f1-score"])
            mlflow.log_metric("precision_ham", report["ham"]["precision"])
            mlflow.log_metric("recall_ham", report["ham"]["recall"])
            mlflow.log_metric("f1_ham", report["ham"]["f1-score"])

            # ---- Confusion Matrix ----
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=sorted(y.unique()),
                yticklabels=sorted(y.unique())
            )
            plt.title("Confusion Matrix - MultinomialNB")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()

            cm_path = f"cm_nb_alpha_{alpha}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # ---- Classification Report ----
            report_path = f"report_nb_alpha_{alpha}.csv"
            pd.DataFrame(report).to_csv(report_path)
            mlflow.log_artifact(report_path)

            # ---- Save Model (MLflow format) ----

            # Log model (MLflow format)
            model_folder = f"model_nb_alpha_{alpha}"
            mlflow.sklearn.log_model(model, artifact_path=model_folder)
            # Setelah log_model, tulis python_version ke artefak model
            model_uri = mlflow.get_artifact_uri(model_folder)
            # Hilangkan prefix file:// jika ada
            if model_uri.startswith("file://"):
                model_dir = urllib.parse.urlparse(model_uri).path
            else:
                model_dir = model_uri
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "python_version"), "w") as f:
                f.write("3.9\n")

            # ---- Cleanup Local Files ----
            os.remove(cm_path)
            os.remove(report_path)

    # ========================================================
    # 6. SVM TUNING (NESTED RUN)
    # ========================================================

    for C in param_grid_svm["C"]:
        with mlflow.start_run(
            run_name=f"SVM_C_{C}",
            nested=True
        ):
            model = SVC(kernel="linear", C=C, probability=True)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            # ---- Manual Logging ----
            mlflow.log_param("model", "SVC")
            mlflow.log_param("C", C)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_spam", report["spam"]["precision"])
            mlflow.log_metric("recall_spam", report["spam"]["recall"])
            mlflow.log_metric("f1_spam", report["spam"]["f1-score"])
            mlflow.log_metric("precision_ham", report["ham"]["precision"])
            mlflow.log_metric("recall_ham", report["ham"]["recall"])
            mlflow.log_metric("f1_ham", report["ham"]["f1-score"])

            # ---- Confusion Matrix ----
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=sorted(y.unique()),
                yticklabels=sorted(y.unique())
            )
            plt.title("Confusion Matrix - SVM")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()

            cm_path = f"cm_svm_C_{C}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # ---- Classification Report ----
            report_path = f"report_svm_C_{C}.csv"
            pd.DataFrame(report).to_csv(report_path)
            mlflow.log_artifact(report_path)

            # ---- Save Model (MLflow format) ----

            # Log model (MLflow format)
            model_folder = f"model_svm_C_{C}"
            mlflow.sklearn.log_model(model, artifact_path=model_folder)
            # Setelah log_model, tulis python_version ke artefak model
            model_uri = mlflow.get_artifact_uri(model_folder)
            # Hilangkan prefix file:// jika ada
            if model_uri.startswith("file://"):
                model_dir = urllib.parse.urlparse(model_uri).path
            else:
                model_dir = model_uri
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "python_version"), "w") as f:
                f.write("3.9\n")

            # ---- Cleanup ----
            os.remove(cm_path)
            os.remove(report_path)

print("Training & logging selesai. Cek MLflow UI & DagsHub.")