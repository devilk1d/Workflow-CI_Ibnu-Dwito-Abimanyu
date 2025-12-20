import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import argparse
import joblib

mlflow.set_experiment("spam-email-classifier")

with mlflow.start_run():
    def train_advance(data_path):
        """
        Training function with MLflow tracking and artifact logging
        """
        # Load Preprocessed Data
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        X = df['clean_text'].astype(str)
        y = df['label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Vectorization
        print("Vectorizing text data...")
        tfidf = TfidfVectorizer(max_features=3000)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # Start MLflow run
        with mlflow.start_run(run_name="RandomForest_CI_Training"):
            print("Starting model training with hyperparameter tuning...")
            
            # Hyperparameter Tuning
            rf = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_tfidf, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Log parameters
            print("Logging parameters...")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("max_features", 3000)
            
            # Log metrics
            print("Logging metrics...")
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            y_pred = best_model.predict(X_test_tfidf)
            test_acc = best_model.score(X_test_tfidf, y_test)
            mlflow.log_metric("test_accuracy", test_acc)
            
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")

            # Create artifacts directory
            os.makedirs("artifacts", exist_ok=True)

            # Artifact 1: Confusion Matrix
            print("Creating confusion matrix...")
            plt.figure(figsize=(8, 6))
            ConfusionMatrixDisplay.from_estimator(
                best_model, X_test_tfidf, y_test, 
                cmap='Blues', values_format='d'
            )
            plt.title("Confusion Matrix - Spam Email Detection")
            cm_path = "artifacts/confusion_matrix.png"
            plt.savefig(cm_path, dpi=100, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(cm_path)

            # Artifact 2: Classification Report
            print("Creating classification report...")
            report = classification_report(y_test, y_pred)
            report_path = "artifacts/classification_report.txt"
            with open(report_path, "w") as f:
                f.write("Classification Report - Spam Email Detection\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
                f.write(f"\n\nBest Parameters: {grid_search.best_params_}\n")
                f.write(f"Best CV Score: {grid_search.best_score_:.4f}\n")
                f.write(f"Test Accuracy: {test_acc:.4f}\n")
            mlflow.log_artifact(report_path)

            # Artifact 3: Save vectorizer
            print("Saving TF-IDF vectorizer...")
            vectorizer_path = "artifacts/tfidf_vectorizer.pkl"
            joblib.dump(tfidf, vectorizer_path)
            mlflow.log_artifact(vectorizer_path)

            # Artifact 4: Model metadata
            print("Creating model metadata...")
            metadata = {
                "model_type": "RandomForestClassifier",
                "best_params": grid_search.best_params_,
                "cv_score": float(grid_search.best_score_),
                "test_accuracy": float(test_acc),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features": 3000
            }
            metadata_path = "artifacts/model_metadata.txt"
            with open(metadata_path, "w") as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            mlflow.log_artifact(metadata_path)

            # Log Model
            print("Logging model to MLflow...")
            mlflow.sklearn.log_model(
                best_model, 
                "spam_model_rf",
                registered_model_name="SpamEmailDetector"
            )
            
            print("\n" + "="*50)
            print("Training completed successfully!")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print("All artifacts logged to DagsHub/MLflow")
            print("="*50)

            return best_model, test_acc

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Train spam email detection model')
        parser.add_argument(
            '--data_path', 
            type=str, 
            default='MLProject/spam_email_dataset_cleaned.csv',
            help='Path to the cleaned dataset'
        )
        
        args = parser.parse_args()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        
        # Train model
        train_advance(args.data_path)