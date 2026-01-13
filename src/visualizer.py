"""
Data Visualizer

Core visualization functions for statistical and business charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


class DataVisualizer:
    """
    Comprehensive data visualization toolkit.

    Parameters
    ----------
    style : str, default='seaborn'
        Plotting style
    figsize : tuple, default=(10, 6)
        Default figure size
    """

    def __init__(self, style='seaborn', figsize=(10, 6)):
        plt.style.use(style if style != 'seaborn' else 'seaborn-v0_8-darkgrid')
        self.default_figsize = figsize
        self.colors = sns.color_palette('husl', 10)

    def plot_distribution(self, data, title='Distribution', xlabel=None,
                         show_stats=True, bins=30, figsize=None):
        """
        Plot distribution with KDE and statistics.

        Parameters
        ----------
        data : array-like
            Data to plot
        title : str
            Plot title
        xlabel : str, optional
            X-axis label
        show_stats : bool
            Show statistics on plot
        bins : int
            Number of histogram bins
        figsize : tuple, optional
            Figure size
        """
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Histogram with KDE
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', density=True, label='Histogram')

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data.dropna())
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Statistics
        if show_stats:
            mean = data.mean()
            median = data.median()
            std = data.std()

            ax.axvline(mean, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
            ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')

            # Stats text
            stats_text = f'μ = {mean:.2f}\nσ = {std:.2f}\nMedian = {median:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(xlabel or 'Value')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_correlation_heatmap(self, data, method='pearson', figsize=(12, 10),
                                annot=True, cmap='coolwarm'):
        """
        Plot correlation heatmap.

        Parameters
        ----------
        data : pd.DataFrame
            Data with numeric columns
        method : str
            Correlation method
        figsize : tuple
            Figure size
        annot : bool
            Show correlation values
        cmap : str
            Color map
        """
        fig, ax = plt.subplots(figsize=figsize)

        corr = data.corr(method=method)

        sns.heatmap(corr, annot=annot, fmt='.2f', cmap=cmap, center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title(f'Correlation Heatmap ({method.title()})', fontsize=14)
        plt.tight_layout()

        return fig

    def plot_timeseries(self, dates, values, title='Time Series',
                       show_trend=False, show_seasonality=False, figsize=None):
        """
        Plot time series with optional trend and seasonality.

        Parameters
        ----------
        dates : array-like
            Date values
        values : array-like
            Time series values
        title : str
            Plot title
        show_trend : bool
            Show trend line
        show_seasonality : bool
            Show seasonal decomposition
        figsize : tuple, optional
            Figure size
        """
        figsize = figsize or (14, 6)

        if show_seasonality:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Create series
            ts = pd.Series(values, index=pd.to_datetime(dates))

            # Decompose
            decomposition = seasonal_decompose(ts, model='additive', period=min(30, len(ts)//2))

            fig, axes = plt.subplots(4, 1, figsize=(14, 12))

            decomposition.observed.plot(ax=axes[0], title='Original')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')

            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(dates, values, linewidth=2, label='Actual')

            if show_trend:
                # Fit polynomial trend
                x_numeric = np.arange(len(dates))
                z = np.polyfit(x_numeric, values, 2)
                p = np.poly1d(z)
                ax.plot(dates, p(x_numeric), 'r--', linewidth=2, label='Trend')

            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

        return fig

    def plot_categorical_comparison(self, categories, values, title='Comparison',
                                   plot_type='bar', figsize=None):
        """
        Plot categorical comparison.

        Parameters
        ----------
        categories : array-like
            Category labels
        values : array-like
            Values for each category
        title : str
            Plot title
        plot_type : str
            Type: 'bar', 'barh', or 'pie'
        figsize : tuple, optional
            Figure size
        """
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)

        if plot_type == 'bar':
            ax.bar(categories, values, color=self.colors, edgecolor='black')
            ax.set_xlabel('Category')
            ax.set_ylabel('Value')
            plt.xticks(rotation=45, ha='right')

        elif plot_type == 'barh':
            ax.barh(categories, values, color=self.colors, edgecolor='black')
            ax.set_xlabel('Value')
            ax.set_ylabel('Category')

        elif plot_type == 'pie':
            ax.pie(values, labels=categories, autopct='%1.1f%%', colors=self.colors,
                  startangle=90)
            ax.axis('equal')

        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y' if plot_type != 'pie' else None)
        plt.tight_layout()

        return fig

    def plot_scatter_with_regression(self, x, y, title='Scatter Plot',
                                    xlabel='X', ylabel='Y', figsize=None):
        """
        Plot scatter with regression line and confidence interval.

        Parameters
        ----------
        x : array-like
            X values
        y : array-like
            Y values
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        figsize : tuple, optional
            Figure size
        """
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(x, y, alpha=0.6, edgecolors='black', s=50)

        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'r-', linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y, p(x))

        # Confidence interval
        predict_mean = p(x)
        residuals = y - predict_mean
        std_error = np.std(residuals)

        ci = 1.96 * std_error
        ax.fill_between(x, predict_mean - ci, predict_mean + ci, alpha=0.2, label='95% CI')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} (R² = {r2:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_box_comparison(self, data, groups, title='Box Plot Comparison', figsize=None):
        """
        Plot box plots for group comparison.

        Parameters
        ----------
        data : array-like
            Data values
        groups : array-like
            Group labels
        title : str
            Plot title
        figsize : tuple, optional
            Figure size
        """
        figsize = figsize or (12, 6)
        fig, ax = plt.subplots(figsize=figsize)

        # Create DataFrame
        df = pd.DataFrame({'value': data, 'group': groups})

        # Box plot
        sns.boxplot(data=df, x='group', y='value', ax=ax, palette='Set2')

        # Add points
        sns.stripplot(data=df, x='group', y='value', ax=ax,
                     color='black', alpha=0.3, size=3)

        ax.set_title(title)
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    def plot_missing_values(self, df, figsize=None):
        """
        Visualize missing values in DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        figsize : tuple, optional
            Figure size
        """
        figsize = figsize or (12, 8)

        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

        if len(missing_pct) == 0:
            print("No missing values found!")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(range(len(missing_pct)), missing_pct.values, color='coral', edgecolor='black')
        ax.set_yticks(range(len(missing_pct)))
        ax.set_yticklabels(missing_pct.index)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Column')
        ax.grid(True, alpha=0.3, axis='x')

        # Add percentage labels
        for i, v in enumerate(missing_pct.values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')

        plt.tight_layout()

        return fig
