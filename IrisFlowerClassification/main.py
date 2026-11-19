"""
End-to-End Execution Pipeline for Iris Flower Classification.
CodeAlpha Data Science Internship - Task 1

Usage:
    python main.py
"""

import sys
import os

# Ensure package root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.data_loader import IrisDataLoader
from src.visualize import IrisVisualizer
from src.model import IrisModelPipeline
from src.predict import predict_species


def run_pipeline():
    print("\n" + "#" * 65)
    print("  CODEALPHA DATA SCIENCE INTERNSHIP: TASK 1 (IRIS CLASSIFICATION)  ")
    print("#" * 65 + "\n")

    # Step 1: Data Loading & Exploration
    print("[1/5] Loading and inspecting dataset...")
    loader = IrisDataLoader()
    df = loader.load_data()
    print("First 5 rows:")
    print(df.head().to_string())
    print("\nStatistical Summary:")
    print(df.describe().round(2).to_string())

    # Step 2: Visualization & EDA
    print("\n[2/5] Generating exploratory visualizations...")
    visualizer = IrisVisualizer()
    visualizer.plot_pairplot(df)
    visualizer.plot_feature_distributions(df)
    visualizer.plot_correlation_heatmap(df)

    # Step 3: Train / Test Split
    print("\n[3/5] Splitting data into stratified train and test partitions...")
    X_train, X_test, y_train, y_test, encoder = loader.get_train_test_split(test_size=0.2, random_state=42)
    class_names = list(encoder.classes_)

    # Step 4: Model Training, Evaluation & Benchmarking
    print("\n[4/5] Training classification models...")
    pipeline = IrisModelPipeline()
    results = pipeline.train_and_evaluate_all(X_train, X_test, y_train, y_test, class_names)

    # Save visual confusion matrix for the best model
    best_name = pipeline.best_model_name
    best_cm = results[best_name]["confusion_matrix"]
    visualizer.plot_confusion_matrix(best_cm, class_names, model_name=best_name)

    # Feature importance if Random Forest trained
    if "Random Forest" in pipeline.trained_models:
        rf_model = pipeline.trained_models["Random Forest"]
        feature_names = [c for c in df.columns if c != "Species"]
        visualizer.plot_feature_importance(feature_names, rf_model.feature_importances_)

    # Save best model
    model_artifact_path = pipeline.save_best_model(encoder, [c for c in df.columns if c != "Species"])

    # Step 5: Verification & Sample Inference
    print("\n[5/5] Running sample validation inference...")
    test_samples = [
        {"desc": "Typical Iris-setosa", "sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2},
        {"desc": "Typical Iris-versicolor", "sl": 5.9, "sw": 3.0, "pl": 4.2, "pw": 1.5},
        {"desc": "Typical Iris-virginica", "sl": 6.7, "sw": 3.3, "pl": 5.7, "pw": 2.1}
    ]

    for sample in test_samples:
        species, probs = predict_species(sample["sl"], sample["sw"], sample["pl"], sample["pw"], model_path=model_artifact_path)
        print(f"Sample [{sample['desc']}]: Sepal({sample['sl']}, {sample['sw']}), Petal({sample['pl']}, {sample['pw']}) => Predicted: {species} (Confidence: {probs.get(species, 0.0)}%)")

    print("\n" + "#" * 65)
    print("  TASK 1 PIPELINE COMPLETED SUCCESSFULLY!  ")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
