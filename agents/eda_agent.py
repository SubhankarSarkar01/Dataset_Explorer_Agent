import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

class EDAAgent:
    """
    Agent responsible for Exploratory Data Analysis:
    - Descriptive statistics (numerical & categorical)
    - Value counts for categorical columns
    - Histograms for numeric columns
    - Boxplots for numeric columns
    - Correlation heatmap for numeric columns
    """
    def __init__(self):
        pass

    def get_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Returns descriptive statistics for numeric and categorical columns.
        """
        numeric_cols = df.select_dtypes(include=[np.number])
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool'])
        
        return {
            "numeric": numeric_cols.describe() if not numeric_cols.empty else pd.DataFrame(),
            "categorical": categorical_cols.describe() if not categorical_cols.empty else pd.DataFrame()
        }

    def get_value_counts(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Returns value counts and percentages for a specific categorical column.
        """
        if column not in df.columns:
            return pd.DataFrame()
        
        # Limit cardinality display to top 20, grouping the rest into 'Other' if necessary
        counts = df[column].value_counts(dropna=False)
        pcts = df[column].value_counts(normalize=True, dropna=False) * 100
        
        val_counts_df = pd.DataFrame({
            "Category Value": counts.index.astype(str),
            "Count": counts.values,
            "Percentage (%)": pcts.values.round(2)
        })
        return val_counts_df

    def plot_histogram(self, df: pd.DataFrame, column: str) -> Optional[go.Figure]:
        """
        Generates an interactive Plotly histogram with a marginal boxplot for a numeric column.
        """
        if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
            return None
        
        # Drop NaNs for plotting
        cleaned_series = df[column].dropna()
        if cleaned_series.empty:
            return None

        fig = px.histogram(
            df, 
            x=column, 
            marginal="box",
            title=f"Distribution & Spread of: {column}",
            color_discrete_sequence=['#636EFA'] # Sleek modern blue
        )
        
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=30, r=30, t=50, b=30),
            height=400,
            xaxis_title=column,
            yaxis_title="Count"
        )
        return fig

    def plot_boxplot(self, df: pd.DataFrame, column: str) -> Optional[go.Figure]:
        """
        Generates an interactive Plotly boxplot for a numeric column.
        """
        if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
            return None
            
        cleaned_series = df[column].dropna()
        if cleaned_series.empty:
            return None

        fig = px.box(
            df,
            y=column,
            title=f"Boxplot Analysis of: {column}",
            color_discrete_sequence=['#EF553B'] # Warm coral red
        )
        
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=30, r=30, t=50, b=30),
            height=400,
            yaxis_title=column
        )
        return fig

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Generates an interactive correlation heatmap for numeric columns.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
            
        corr_matrix = numeric_df.corr(method='pearson').round(2)
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r", # Diverging red-blue color scale
            zmin=-1.0,
            zmax=1.0,
            title="Pearson Correlation Heatmap"
        )
        
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=30, r=30, t=50, b=30),
            height=500
        )
        return fig
