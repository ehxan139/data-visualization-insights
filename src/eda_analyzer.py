"""
Automated Exploratory Data Analysis (EDA)

Comprehensive EDA with automated insights and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class EDAAnalyzer:
    """
    Automated EDA analyzer for pandas DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    sample_size : int, optional
        Sample size for large datasets
    """

    def __init__(self, df, sample_size=None):
        self.df = df.copy()

        # Sample if needed
        if sample_size and len(df) > sample_size:
            self.df = df.sample(sample_size, random_state=42)
            print(f"Sampled {sample_size} rows from {len(df)} total rows")

        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    def get_summary_statistics(self):
        """
        Get comprehensive summary statistics.

        Returns
        -------
        summary : dict
            Dataset summary
        """
        summary = {
            'shape': self.df.shape,
            'num_numeric': len(self.numeric_cols),
            'num_categorical': len(self.categorical_cols),
            'num_datetime': len(self.datetime_cols),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum()
        }

        return summary

    def analyze_missing_values(self):
        """
        Analyze missing values in dataset.

        Returns
        -------
        missing_info : pd.DataFrame
            Missing value statistics per column
        """
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100

        missing_info = pd.DataFrame({
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct
        })

        missing_info = missing_info[missing_info['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )

        return missing_info

    def identify_outliers(self, method='iqr', threshold=1.5):
        """
        Identify outliers in numeric columns.

        Parameters
        ----------
        method : str
            Detection method: 'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection

        Returns
        -------
        outliers : dict
            Outlier counts per column
        """
        outliers = {}

        for col in self.numeric_cols:
            values = self.df[col].dropna()

            if method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outlier_mask = (values < lower) | (values > upper)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(values))
                outlier_mask = z_scores > threshold

            outliers[col] = {
                'count': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(values)) * 100
            }

        return outliers

    def analyze_distributions(self):
        """
        Analyze distributions of numeric columns.

        Returns
        -------
        dist_info : pd.DataFrame
            Distribution statistics
        """
        dist_info = []

        for col in self.numeric_cols:
            values = self.df[col].dropna()

            # Skewness and kurtosis
            skew = stats.skew(values)
            kurt = stats.kurtosis(values)

            # Normality test
            _, p_value = stats.normaltest(values) if len(values) > 8 else (0, 1)

            dist_info.append({
                'column': col,
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'skewness': skew,
                'kurtosis': kurt,
                'is_normal': p_value > 0.05
            })

        return pd.DataFrame(dist_info)

    def analyze_correlations(self, method='pearson', threshold=0.7):
        """
        Analyze correlations between numeric columns.

        Parameters
        ----------
        method : str
            Correlation method: 'pearson', 'spearman', or 'kendall'
        threshold : float
            Threshold for strong correlations

        Returns
        -------
        strong_corr : pd.DataFrame
            Strongly correlated pairs
        """
        corr_matrix = self.df[self.numeric_cols].corr(method=method)

        # Extract strong correlations
        strong_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        return pd.DataFrame(strong_pairs).sort_values('correlation', ascending=False, key=abs)

    def analyze_categorical_variables(self):
        """
        Analyze categorical variables.

        Returns
        -------
        cat_info : pd.DataFrame
            Categorical variable statistics
        """
        cat_info = []

        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()

            cat_info.append({
                'column': col,
                'unique_values': len(value_counts),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_common_freq': value_counts.iloc[-1] if len(value_counts) > 0 else 0
            })

        return pd.DataFrame(cat_info)

    def identify_data_issues(self):
        """
        Identify potential data quality issues.

        Returns
        -------
        issues : list of dict
            List of identified issues
        """
        issues = []

        # Missing values
        missing = self.analyze_missing_values()
        if len(missing) > 0:
            issues.append({
                'type': 'missing_values',
                'severity': 'high',
                'description': f"{len(missing)} columns have missing values",
                'details': missing.to_dict()
            })

        # Duplicates
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            issues.append({
                'type': 'duplicates',
                'severity': 'medium',
                'description': f"{dup_count} duplicate rows found",
                'details': {'count': dup_count}
            })

        # High cardinality categoricals
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            if unique_count > 50:
                issues.append({
                    'type': 'high_cardinality',
                    'severity': 'low',
                    'description': f"Column '{col}' has {unique_count} unique values",
                    'details': {'column': col, 'unique_count': unique_count}
                })

        # Constant columns
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_cols:
            issues.append({
                'type': 'constant_columns',
                'severity': 'medium',
                'description': f"{len(constant_cols)} columns have constant values",
                'details': {'columns': constant_cols}
            })

        return issues

    def generate_full_report(self, output_dir='eda_results/'):
        """
        Generate comprehensive EDA report with visualizations.

        Parameters
        ----------
        output_dir : str
            Output directory for report files

        Returns
        -------
        report : dict
            Summary report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Generating EDA report...")

        # Summary statistics
        print("  - Summary statistics")
        summary = self.get_summary_statistics()

        # Missing values
        print("  - Missing value analysis")
        missing = self.analyze_missing_values()

        # Distributions
        print("  - Distribution analysis")
        distributions = self.analyze_distributions()

        # Correlations
        print("  - Correlation analysis")
        correlations = self.analyze_correlations()

        # Categorical analysis
        print("  - Categorical analysis")
        categorical = self.analyze_categorical_variables()

        # Data issues
        print("  - Data quality issues")
        issues = self.identify_data_issues()

        # Save reports
        with open(f"{output_dir}/summary.txt", 'w') as f:
            f.write("DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

            f.write("\n\nDATA QUALITY ISSUES\n")
            f.write("=" * 50 + "\n\n")
            for issue in issues:
                f.write(f"[{issue['severity'].upper()}] {issue['description']}\n")

        if len(missing) > 0:
            missing.to_csv(f"{output_dir}/missing_values.csv")

        distributions.to_csv(f"{output_dir}/distributions.csv", index=False)

        if len(correlations) > 0:
            correlations.to_csv(f"{output_dir}/correlations.csv", index=False)

        categorical.to_csv(f"{output_dir}/categorical_analysis.csv", index=False)

        print(f"\nReport saved to {output_dir}")

        return {
            'summary': summary,
            'missing_values': missing,
            'distributions': distributions,
            'correlations': correlations,
            'categorical': categorical,
            'issues': issues
        }
