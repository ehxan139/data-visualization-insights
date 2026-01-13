"""
Utility functions for data visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def set_plot_style(style='seaborn', context='notebook', palette='husl'):
    """
    Set global plotting style.

    Parameters
    ----------
    style : str
        Matplotlib style
    context : str
        Seaborn context: 'paper', 'notebook', 'talk', 'poster'
    palette : str
        Color palette
    """
    plt.style.use(style if style != 'seaborn' else 'seaborn-v0_8-darkgrid')
    sns.set_context(context)
    sns.set_palette(palette)


def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Save figure with high quality.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    dpi : int
        Resolution
    bbox_inches : str
        Bounding box setting
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved to {filename}")


def create_subplot_grid(n_plots, ncols=3, figsize_per_plot=(5, 4)):
    """
    Create subplot grid for multiple plots.

    Parameters
    ----------
    n_plots : int
        Number of plots
    ncols : int
        Number of columns
    figsize_per_plot : tuple
        Size of each subplot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : array
        Array of axes
    """
    nrows = (n_plots + ncols - 1) // ncols
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Hide extra subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    return fig, axes


def format_large_numbers(x, pos=None):
    """
    Format large numbers for axis labels.

    Parameters
    ----------
    x : float
        Number to format
    pos : int, optional
        Position (required by FuncFormatter)

    Returns
    -------
    formatted : str
        Formatted string
    """
    if abs(x) >= 1e9:
        return f'{x/1e9:.1f}B'
    elif abs(x) >= 1e6:
        return f'{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.1f}K'
    else:
        return f'{x:.0f}'


def add_value_labels(ax, spacing=5, format_str='.0f'):
    """
    Add value labels to bar chart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object
    spacing : int
        Space between bar and label
    format_str : str
        Number format string
    """
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        label = f'{y_value:{format_str}}'

        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, spacing),
            textcoords="offset points",
            ha='center',
            va='bottom'
        )


def create_custom_colormap(colors):
    """
    Create custom colormap from list of colors.

    Parameters
    ----------
    colors : list
        List of color strings

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Custom colormap
    """
    from matplotlib.colors import LinearSegmentedColormap

    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    return cmap


def export_plot_data(fig, filename='plot_data.csv'):
    """
    Export plot data to CSV.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    filename : str
        Output filename
    """
    all_data = []

    for ax in fig.get_axes():
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            label = line.get_label()

            df = pd.DataFrame({
                f'{label}_x': x_data,
                f'{label}_y': y_data
            })
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, axis=1)
        combined.to_csv(filename, index=False)
        print(f"Plot data exported to {filename}")


def annotate_outliers(ax, x, y, threshold=3):
    """
    Annotate outlier points on scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object
    x : array-like
        X values
    y : array-like
        Y values
    threshold : float
        Z-score threshold for outliers
    """
    from scipy import stats

    z_scores = np.abs(stats.zscore(y))
    outliers = z_scores > threshold

    for i, (xi, yi, is_outlier) in enumerate(zip(x, y, outliers)):
        if is_outlier:
            ax.annotate(
                f'{yi:.1f}',
                (xi, yi),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='red'
            )
