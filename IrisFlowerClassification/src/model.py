"""
Model Training, Benchmark, and Evaluation for Iris Flower Classification.
CodeAlpha Data Science Internship - Task 1
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class IrisModelPipeline:
    """Manages training, multi-algorithm benchmarking, and serialization."""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.evaluation_results = {}
        self.best_model_name = None

    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test, class_names):
        """Trains Logistic Regression, KNN, and Random Forest, benchmarked on test data."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        candidate_models = {
            "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        best_acc = -1.0

        print("=" * 60)
        print("          IRIS FLOWER CLASSIFICATION BENCHMARK")
        print("=" * 60)

        for name, clf in candidate_models.items():
            # For tree-based models like Random Forest, scaled or unscaled are both fine; using scaled for consistency
            clf.fit(X_train_scaled, y_train)
            self.trained_models[name] = clf

            y_pred_train = clf.predict(X_train_scaled)
            y_pred_test = clf.predict(X_test_scaled)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            cm = confusion_matrix(y_test, y_pred_test)
            report = classification_report(y_test, y_pred_test, target_names=class_names, output_dict=True)

            self.evaluation_results[name] = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "confusion_matrix": cm,
                "classification_report": report
            }

            print(f"\n--- {name} ---")
            print(f"Training Accuracy : {train_acc * 100:.2f}%")
            print(f"Test Accuracy     : {test_acc * 100:.2f}%")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_test, target_names=class_names))

            if test_acc > best_acc:
                best_acc = test_acc
                self.best_model_name = name

        print("=" * 60)
        print(f"Best Performing Model: {self.best_model_name} with Test Accuracy: {best_acc * 100:.2f}%")
        print("=" * 60)
        return self.evaluation_results

    def save_best_model(self, label_encoder, feature_names):
        """Saves best model, scaler, and metadata into a joblib bundle."""
        if not self.best_model_name:
            raise ValueError("No model has been trained yet!")

        artifact = {
            "model_name": self.best_model_name,
            "model": self.trained_models[self.best_model_name],
            "scaler": self.scaler,
            "label_encoder": label_encoder,
            "feature_names": list(feature_names),
            "classes": list(label_encoder.classes_)
        }
        save_path = os.path.join(self.models_dir, "iris_best_model.joblib")
        joblib.dump(artifact, save_path)
        print(f"Serialized best model bundle saved to: {save_path}")
        return save_path
