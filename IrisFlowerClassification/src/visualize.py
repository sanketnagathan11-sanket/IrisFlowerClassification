"""
Visualization Module for Iris Flower Classification.
Generates informative, publication-grade exploratory and evaluation plots.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class IrisVisualizer:
    """Creates and saves aesthetic visualizations for EDA and model evaluation."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "visuals")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def plot_pairplot(self, df):
        """Generates pairplot highlighting distinct cluster separability across species."""
        save_path = os.path.join(self.output_dir, "iris_pairplot.png")
        pair_grid = sns.pairplot(df, hue="Species", markers=["o", "s", "D"], height=2.4)
        pair_grid.fig.suptitle("Iris Feature Pairplot - Species Separability", y=1.02, fontsize=14, weight="bold")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_feature_distributions(self, df):
        """Plots boxplots to visualize measurement ranges and outlier characteristics."""
        save_path = os.path.join(self.output_dir, "feature_boxplots.png")
        feature_cols = [c for c in df.columns if c != "Species"]

        plt.figure(figsize=(12, 6))
        for i, col in enumerate(feature_cols, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(data=df, x="Species", y=col, hue="Species", legend=False)
            plt.title(f"Distribution of {col}", fontsize=11, weight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_correlation_heatmap(self, df):
        """Plots correlation matrix between morphological dimensions."""
        save_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        feature_cols = [c for c in df.columns if c != "Species"]
        corr = df[feature_cols].corr()

        plt.figure(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.8, vmin=-1, vmax=1)
        plt.title("Iris Morphological Feature Correlation Matrix", fontsize=12, weight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_confusion_matrix(self, cm, class_names, model_name: str = "Classifier"):
        """Plots and saves normalized or raw confusion matrix heatmap."""
        save_path = os.path.join(self.output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, linewidths=0.5)
        plt.title(f"Confusion Matrix - {model_name}", fontsize=12, weight="bold")
        plt.xlabel("Predicted Species", fontsize=11, weight="bold")
        plt.ylabel("Actual Species", fontsize=11, weight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_feature_importance(self, feature_names, importances, model_name: str = "Random Forest"):
        """Plots relative feature importance ranking."""
        save_path = os.path.join(self.output_dir, "feature_importance.png")
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(8, 4))
        plt.bar(range(len(importances)), [importances[i] for i in indices], color="#3498db", edgecolor="#2980b9")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=15, weight="bold")
        plt.title(f"Feature Importance Ranking ({model_name})", fontsize=12, weight="bold")
        plt.ylabel("Relative Gini Importance", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
