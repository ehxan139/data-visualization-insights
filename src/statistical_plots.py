"""
Statistical Visualization Utilities

Specialized statistical plots for analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


def plot_qq(data, title='Q-Q Plot', figsize=(8, 6)):
    """
    Plot Q-Q plot for normality assessment.
    
    Parameters
    ----------
    data : array-like
        Data to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_residuals(y_true, y_pred, figsize=(12, 5)):
    """
    Plot residual diagnostics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    figsize : tuple
        Figure size
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs fitted
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_pairplot(df, hue=None, figsize=(12, 12)):
    """
    Create pair plot for multivariate analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with numeric columns
    hue : str, optional
        Column for color grouping
    figsize : tuple
        Figure size
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if hue and hue in numeric_cols:
        numeric_cols.remove(hue)
    
    # Limit to 5 columns for readability
    if len(numeric_cols) > 5:
        numeric_cols = numeric_cols[:5]
    
    plot_data = df[numeric_cols + ([hue] if hue else [])]
    
    g = sns.pairplot(plot_data, hue=hue, diag_kind='kde', plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pair Plot', y=1.01)
    
    return g.fig


def plot_violin_comparison(data, groups, title='Violin Plot', figsize=(10, 6)):
    """
    Plot violin plots for distribution comparison.
    
    Parameters
    ----------
    data : array-like
        Data values
    groups : array-like
        Group labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df = pd.DataFrame({'value': data, 'group': groups})
    
    sns.violinplot(data=df, x='group', y='value', ax=ax, palette='muted')
    
    ax.set_title(title)
    ax.set_xlabel('Group')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def plot_kde_comparison(data_dict, title='KDE Comparison', figsize=(10, 6)):
    """
    Plot KDE curves for multiple distributions.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of {label: data} for each distribution
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for label, data in data_dict.items():
        data = pd.Series(data).dropna()
        data.plot(kind='kde', ax=ax, label=label, linewidth=2)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_ecdf(data, title='Empirical CDF', figsize=(8, 6)):
    """
    Plot empirical cumulative distribution function.
    
    Parameters
    ----------
    data : array-like
        Data values
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    data_sorted = np.sort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)
    
    ax.plot(data_sorted, ecdf, linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_confidence_intervals(means, std_errors, labels, title='Confidence Intervals', figsize=(10, 6)):
    """
    Plot confidence intervals for multiple groups.
    
    Parameters
    ----------
    means : array-like
        Mean values
    std_errors : array-like
        Standard errors
    labels : list
        Group labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ci = 1.96 * np.array(std_errors)  # 95% CI
    
    x_pos = np.arange(len(labels))
    ax.errorbar(x_pos, means, yerr=ci, fmt='o', markersize=8, 
               capsize=5, capthick=2, linewidth=2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return fig
