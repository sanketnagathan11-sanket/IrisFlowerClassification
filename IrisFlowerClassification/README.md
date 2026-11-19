# - Iris Flower Classification (Task 1)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

---

## 📌 Project Overview
This repository contains the complete implementation for **Task 1: Iris Flower Classification** as part of the **CodeAlpha Data Science Internship**.

The goal of this project is to build and evaluate machine learning models capable of classifying Iris flowers into one of three species (*Iris-setosa*, *Iris-versicolor*, or *Iris-virginica*) based on physical morphological measurements:
1. **Sepal Length (cm)**
2. **Sepal Width (cm)**
3. **Petal Length (cm)**
4. **Petal Width (cm)**

---

## 📂 Project Architecture

```
IrisFlowerClassification/
│
├── data/
│   └── iris.csv                            # Raw Iris dataset (150 samples)
├── models/
│   └── iris_best_model.joblib              # Serialized best model & scaler artifact
├── notebooks/
│   └── iris_flower_classification.ipynb    # Interactive step-by-step notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                      # Data ingestion & stratified train/test split
│   ├── model.py                            # Model training, benchmarking & serialization
│   ├── predict.py                          # Interactive CLI prediction tool
│   └── visualize.py                        # EDA & evaluation visualization generator
├── visuals/
│   ├── iris_pairplot.png                   # Feature pairplot colored by species
│   ├── feature_boxplots.png                # Boxplots showing distributions & outliers
│   ├── correlation_heatmap.png             # Correlation heatmap of features
│   ├── logistic_regression_confusion_matrix.png
│   └── feature_importance.png              # Relative Gini feature importances
├── main.py                                 # Automated pipeline runner
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/IrisFlowerClassification.git
   cd IrisFlowerClassification
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Run the Full Automated Pipeline
To execute data loading, exploratory analysis, model training, evaluation, plot generation, and validation inference:
```bash
python main.py
```

### 2. Interactive Flower Species Prediction
Use the inference tool with CLI flags or interactively:
```bash
python src/predict.py --sepal_len 5.1 --sepal_wid 3.5 --petal_len 1.4 --petal_wid 0.2
```
Example Output:
```
=============================================
           PREDICTION RESULT
=============================================
Measurements: Sepal(5.1x3.5 cm), Petal(1.4x0.2 cm)
Predicted Species:  >>> Iris-setosa <<<

Class Probabilities:
  - Iris-setosa     :  98.40% | ###################
  - Iris-versicolor :   1.50% | 
  - Iris-virginica  :   0.10% | 
=============================================
```

---

## 📊 Exploratory Data Analysis & Key Insights
- **Linearly Separable Species**: *Iris-setosa* is distinctly separated from *Iris-versicolor* and *Iris-virginica* solely using petal measurements.
- **Strong Feature Importance**: **Petal Length** and **Petal Width** exhibit the highest Gini importance and correlate heavily with species divergence ($r > 0.8$).

---

## 📈 Model Comparison & Benchmark Results

| Model | Training Accuracy | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Multinomial)** | 97.50% | **96.67%** | 0.97 | 0.97 | **0.97** |
| **K-Nearest Neighbors ($k=5$)** | 96.67% | **96.67%** | 0.97 | 0.97 | **0.97** |
| **Random Forest (100 Trees)** | 100.00% | **96.67%** | 0.97 | 0.97 | **0.97** |

---

## 👨‍💻 Author & Acknowledgements
- **Author**: Sanket Nagatghan (CodeAlpha Data Science Intern)
- **Organization**: [CodeAlpha](https://www.codealpha.tech)
