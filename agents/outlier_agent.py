import pandas as pd
import numpy as np
from typing import Dict, Any

class OutlierAgent:
    """
    Agent responsible for detecting outliers in numeric columns:
    - IQR Method: Values outside [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    - Z-Score Method: Values standardizing to absolute scores greater than 3.0
    """
    def __init__(self):
        pass

    def analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects outliers in all numeric columns using both IQR and Z-Score.
        Returns a dictionary summarizing findings for every numeric column.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        column_summaries = {}
        total_iqr_outliers = 0
        total_z_outliers = 0
        
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
                
            # IQR Method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_iqr = q1 - 1.5 * iqr
            upper_iqr = q3 + 1.5 * iqr
            
            iqr_outliers = series[(series < lower_iqr) | (series > upper_iqr)]
            iqr_outliers_count = len(iqr_outliers)
            total_iqr_outliers += iqr_outliers_count
            
            # Z-Score Method
            mean = series.mean()
            std = series.std()
            
            if std > 0:
                z_scores = (series - mean) / std
                z_outliers = series[np.abs(z_scores) > 3.0]
                z_outliers_count = len(z_outliers)
            else:
                z_outliers_count = 0
                
            total_z_outliers += z_outliers_count
            
            column_summaries[col] = {
                "iqr_count": iqr_outliers_count,
                "iqr_bounds": (round(float(lower_iqr), 3), round(float(upper_iqr), 3)),
                "zscore_count": z_outliers_count,
                "mean": round(float(mean), 3),
                "std": round(float(std), 3) if std > 0 else 0.0
            }
            
        return {
            "columns": column_summaries,
            "total_iqr_outliers": total_iqr_outliers,
            "total_zscore_outliers": total_z_outliers,
            "num_numeric_columns": len(numeric_cols)
        }
