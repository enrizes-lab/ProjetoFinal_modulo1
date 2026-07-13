import matplotlib.pyplot as plt, seaborn as sns, os, numpy as np
from src.config import FIGURES_DIR

def plot_target_distribution(df):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['popularity'], kde=True, bins=30, color='mediumpurple')
    plt.savefig(os.path.join(FIGURES_DIR, '01_distribuicao.png'))
    plt.show()

def plot_correlation_heatmap(df):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.savefig(os.path.join(FIGURES_DIR, '02_correlacao.png'))
    plt.show()