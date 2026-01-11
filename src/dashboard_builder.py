"""
Interactive Dashboard Builder

Create interactive dashboards with Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class DashboardBuilder:
    """
    Build interactive dashboards with Plotly.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    title : str, default='Interactive Dashboard'
        Dashboard title
    """
    
    def __init__(self, df, title='Interactive Dashboard'):
        self.df = df
        self.title = title
        self.figures = []
    
    def add_kpi_cards(self, metrics, prefix='', suffix=''):
        """
        Add KPI metric cards.
        
        Parameters
        ----------
        metrics : list or dict
            Metric names or {name: value} dict
        prefix : str
            Prefix for values (e.g., '$')
        suffix : str
            Suffix for values (e.g., '%')
        """
        if isinstance(metrics, list):
            metric_values = {m: self.df[m].sum() if m in self.df.columns else 0 for m in metrics}
        else:
            metric_values = metrics
        
        fig = go.Figure()
        
        for i, (name, value) in enumerate(metric_values.items()):
            fig.add_trace(go.Indicator(
                mode='number',
                value=value,
                title={'text': name.replace('_', ' ').title()},
                number={'prefix': prefix, 'suffix': suffix},
                domain={'x': [i/len(metric_values), (i+1)/len(metric_values)], 'y': [0, 1]}
            ))
        
        fig.update_layout(
            title='Key Performance Indicators',
            height=200,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        self.figures.append(fig)
        return fig
    
    def add_timeseries_chart(self, date_col, value_col, title='Time Series'):
        """
        Add interactive time series chart.
        
        Parameters
        ----------
        date_col : str
            Date column name
        value_col : str
            Value column name
        title : str
            Chart title
        """
        fig = px.line(
            self.df,
            x=date_col,
            y=value_col,
            title=title,
            labels={date_col: 'Date', value_col: value_col.replace('_', ' ').title()}
        )
        
        fig.update_traces(mode='lines+markers')
        fig.update_layout(hovermode='x unified')
        
        self.figures.append(fig)
        return fig
    
    def add_categorical_breakdown(self, category_col, value_col, 
                                  chart_type='bar', title='Breakdown'):
        """
        Add categorical breakdown chart.
        
        Parameters
        ----------
        category_col : str
            Category column
        value_col : str
            Value column
        chart_type : str
            Chart type: 'bar', 'pie', or 'treemap'
        title : str
            Chart title
        """
        grouped = self.df.groupby(category_col)[value_col].sum().reset_index()
        
        if chart_type == 'bar':
            fig = px.bar(
                grouped,
                x=category_col,
                y=value_col,
                title=title
            )
        elif chart_type == 'pie':
            fig = px.pie(
                grouped,
                names=category_col,
                values=value_col,
                title=title
            )
        elif chart_type == 'treemap':
            fig = px.treemap(
                grouped,
                path=[category_col],
                values=value_col,
                title=title
            )
        
        self.figures.append(fig)
        return fig
    
    def add_scatter_plot(self, x_col, y_col, color_col=None, title='Scatter Plot'):
        """
        Add interactive scatter plot.
        
        Parameters
        ----------
        x_col : str
            X-axis column
        y_col : str
            Y-axis column
        color_col : str, optional
            Color grouping column
        title : str
            Chart title
        """
        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            trendline='ols'
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        
        self.figures.append(fig)
        return fig
    
    def add_distribution_plot(self, column, title='Distribution'):
        """
        Add distribution histogram with KDE.
        
        Parameters
        ----------
        column : str
            Column name
        title : str
            Chart title
        """
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=self.df[column],
            name='Histogram',
            opacity=0.7,
            histnorm='probability density'
        ))
        
        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(self.df[column].dropna())
        x_range = np.linspace(self.df[column].min(), self.df[column].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            name='KDE',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=column.replace('_', ' ').title(),
            yaxis_title='Density',
            hovermode='x'
        )
        
        self.figures.append(fig)
        return fig
    
    def add_heatmap(self, columns=None, title='Correlation Heatmap'):
        """
        Add correlation heatmap.
        
        Parameters
        ----------
        columns : list, optional
            Columns to include
        title : str
            Chart title
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr = self.df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={'size': 10}
        ))
        
        fig.update_layout(
            title=title,
            height=600
        )
        
        self.figures.append(fig)
        return fig
    
    def add_funnel_chart(self, stages, values, title='Conversion Funnel'):
        """
        Add funnel chart for conversion analysis.
        
        Parameters
        ----------
        stages : list
            Stage names
        values : list
            Values for each stage
        title : str
            Chart title
        """
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textinfo='value+percent previous'
        ))
        
        fig.update_layout(title=title)
        
        self.figures.append(fig)
        return fig
    
    def export_html(self, filename='dashboard.html'):
        """
        Export dashboard to HTML file.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            f.write(f'<html><head><title>{self.title}</title></head><body>\n')
            f.write(f'<h1 style="text-align:center">{self.title}</h1>\n')
            
            for i, fig in enumerate(self.figures):
                f.write(f'<div id="chart_{i}"></div>\n')
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            
            f.write('</body></html>')
        
        print(f"Dashboard exported to {filename}")
    
    def show_all(self):
        """Display all charts."""
        for fig in self.figures:
            fig.show()
