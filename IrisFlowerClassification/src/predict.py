"""
Interactive CLI Prediction Module for Iris Flower Classification.
CodeAlpha Data Science Internship - Task 1
"""

import os
import argparse
import joblib
import numpy as np


def load_inference_artifact(model_path: str = None):
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "iris_best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model artifact not found at {model_path}. Run main.py first.")
    return joblib.load(model_path)


def predict_species(sepal_len: float, sepal_wid: float, petal_len: float, petal_wid: float, model_path: str = None):
    """Predicts iris species and class probabilities given 4 measurements."""
    artifact = load_inference_artifact(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    encoder = artifact["label_encoder"]
    classes = artifact["classes"]

    raw_input = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    scaled_input = scaler.transform(raw_input)

    pred_idx = model.predict(scaled_input)[0]
    predicted_species = encoder.inverse_transform([pred_idx])[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_input)[0]
        for cls_name, prob in zip(classes, probs):
            probabilities[cls_name] = round(float(prob) * 100, 2)

    return predicted_species, probabilities


def main():
    parser = argparse.ArgumentParser(description="Classify Iris species from flower measurements.")
    parser.add_argument("--sepal_len", type=float, default=None, help="Sepal Length (cm)")
    parser.add_argument("--sepal_wid", type=float, default=None, help="Sepal Width (cm)")
    parser.add_argument("--petal_len", type=float, default=None, help="Petal Length (cm)")
    parser.add_argument("--petal_wid", type=float, default=None, help="Petal Width (cm)")

    args = parser.parse_args()

    # Prompt user if arguments omitted
    if None in (args.sepal_len, args.sepal_wid, args.petal_len, args.petal_wid):
        print("\n--- Iris Flower Species Classifier ---")
        print("Enter flower measurements in centimeters:")
        sl = float(input("Sepal Length (cm) [e.g. 5.1]: ") or 5.1)
        sw = float(input("Sepal Width  (cm) [e.g. 3.5]: ") or 3.5)
        pl = float(input("Petal Length (cm) [e.g. 1.4]: ") or 1.4)
        pw = float(input("Petal Width  (cm) [e.g. 0.2]: ") or 0.2)
    else:
        sl, sw, pl, pw = args.sepal_len, args.sepal_wid, args.petal_len, args.petal_wid

    species, probs = predict_species(sl, sw, pl, pw)

    print("\n" + "=" * 45)
    print("           PREDICTION RESULT")
    print("=" * 45)
    print(f"Measurements: Sepal({sl}x{sw} cm), Petal({pl}x{pw} cm)")
    print(f"Predicted Species:  >>> {species} <<<")
    if probs:
        print("\nClass Probabilities:")
        for c, p in probs.items():
            bar = "#" * int(p // 5)
            print(f"  - {c:<16}: {p:6.2f}% | {bar}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
