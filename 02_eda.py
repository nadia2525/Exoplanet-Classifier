"""
Exploratory data analysis on the cleaned KOI dataset.

Run: python 02_eda.py
Output: plots saved to figures/
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/koi_clean.csv"
FIG_DIR = "figures"

FEATURE_COLS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr",
    "koi_steff", "koi_slogg", "koi_srad",
]

os.makedirs(FIG_DIR, exist_ok=True)
sns.set_style("whitegrid")


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df[FEATURE_COLS + ["label"]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature correlation with confirmed-planet label")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Correlation with label:\n{corr['label'].sort_values(ascending=False)}")


def plot_feature_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        sns.boxplot(data=df, x="label", y=col, ax=ax)
        ax.set_title(col)
        ax.set_xticklabels(["False Positive", "Confirmed"])
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_distributions.png", dpi=150)
    plt.close()


def plot_snr_vs_depth(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="koi_model_snr", y="koi_depth",
        hue="label", alpha=0.5, palette={0: "tomato", 1: "steelblue"},
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Transit SNR vs. depth, by classification")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/snr_vs_depth.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_snr_vs_depth(df)
    print(f"\nSaved plots to {FIG_DIR}/")
