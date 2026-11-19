"""
Data Loading and Preprocessing Module for Iris Flower Classification.
CodeAlpha Data Science Internship - Task 1
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class IrisDataLoader:
    """Handles loading, cleaning, and partitioning of the Iris flower dataset."""

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "iris.csv")
        self.data_path = data_path
        self.df = None
        self.label_encoder = LabelEncoder()

    def load_data(self) -> pd.DataFrame:
        """Loads data from CSV, validates schema, and removes unnecessary identifiers."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Iris dataset file not found at: {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # Drop Id column if present
        if "Id" in self.df.columns:
            self.df = self.df.drop(columns=["Id"])

        # Validate non-emptiness and no missing values
        null_counts = self.df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"Warning: Found missing values:\n{null_counts[null_counts > 0]}")
            self.df = self.df.dropna()

        print(f"Dataset successfully loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        print(f"Target class distribution:\n{self.df['Species'].value_counts().to_string()}\n")
        return self.df

    def get_features_and_target(self):
        """Extracts feature matrix X and encoded target vector y."""
        if self.df is None:
            self.load_data()

        feature_cols = [c for c in self.df.columns if c != "Species"]
        X = self.df[feature_cols]
        y_raw = self.df["Species"]

        y = self.label_encoder.fit_transform(y_raw)
        return X, y, self.label_encoder

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Provides stratified train/test split to guarantee balanced class distributions."""
        X, y, encoder = self.get_features_and_target()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Stratified Split: Train set ({len(X_train)} samples), Test set ({len(X_test)} samples).")
        return X_train, X_test, y_train, y_test, encoder
