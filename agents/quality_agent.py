import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DataQualityAgent:
    """
    Agent responsible for evaluating dataset quality:
    - Count duplicates
    - Count missing values and percentages
    - Detect incorrect data types (e.g., date/numeric columns stored as objects)
    - Provide a quality score and summary
    """
    def __init__(self):
        pass

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs data quality analysis on the DataFrame.
        """
        num_rows, num_cols = df.shape
        
        if num_rows == 0:
            return {
                "duplicate_count": 0,
                "duplicate_percentage": 0.0,
                "missing_info": [],
                "total_missing": 0,
                "total_missing_percentage": 0.0,
                "incorrect_dtypes": [],
                "quality_score": 0.0,
                "quality_grade": "Empty Dataset"
            }

        # 1. Duplicates
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = (duplicate_count / num_rows * 100)

        # 2. Missing values
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / num_rows * 100)
        
        missing_info = []
        for col in df.columns:
            missing_info.append({
                "column": col,
                "missing_count": int(missing_counts[col]),
                "missing_percentage": float(missing_pcts[col]),
                "dtype": str(df[col].dtype)
            })
            
        total_cells = num_rows * num_cols
        total_missing = int(missing_counts.sum())
        total_missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

        # 3. Detect incorrect data types
        incorrect_dtypes = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            
            # Skip if column is already numeric or datetime
            if np.issubdtype(df[col].dtype, np.number) or np.issubdtype(df[col].dtype, np.datetime64):
                continue
                
            # Sample up to 100 non-null values to test formats
            non_null_vals = df[col].dropna()
            if non_null_vals.empty:
                continue
                
            sample_vals = non_null_vals.head(100)
            
            numeric_parse_success = 0
            for val in sample_vals:
                val_str = str(val).strip()
                # Clean common symbols for financial or percentage representation
                cleaned = val_str.replace('$', '').replace('%', '').replace(',', '').strip()
                # Handle potential negative values in parentheses e.g. (100) -> -100
                if cleaned.startswith('(') and cleaned.endswith(')'):
                    cleaned = '-' + cleaned[1:-1]
                try:
                    float(cleaned)
                    numeric_parse_success += 1
                except ValueError:
                    pass
            
            datetime_parse_success = 0
            for val in sample_vals:
                val_str = str(val).strip()
                # Skip numeric-looking values or very short strings to prevent false positives
                if val_str.replace('.', '', 1).isdigit() or len(val_str) < 6:
                    continue
                try:
                    pd.to_datetime(val_str, errors='raise')
                    datetime_parse_success += 1
                except (ValueError, TypeError, OverflowError):
                    pass

            sample_len = len(sample_vals)
            if sample_len > 0:
                # If a large majority parse as numeric or datetime, recommend correct dtype
                if (numeric_parse_success / sample_len) > 0.8:
                    incorrect_dtypes.append({
                        "column": col,
                        "current_type": dtype_str,
                        "suggested_type": "numeric",
                        "reason": f"Over 80% of sampled values ({numeric_parse_success}/{sample_len}) can be parsed as numbers after stripping symbols like $, %."
                    })
                elif (datetime_parse_success / sample_len) > 0.8:
                    incorrect_dtypes.append({
                        "column": col,
                        "current_type": dtype_str,
                        "suggested_type": "datetime",
                        "reason": f"Over 80% of sampled values ({datetime_parse_success}/{sample_len}) match common date/time formats."
                    })

        # 4. Score Calculation
        # missing values penalty (up to 40 pts): 1% missing total cells = 2 pts penalty
        missing_penalty = min(total_missing_pct * 2, 40)
        # duplicate rows penalty (up to 30 pts): 1% duplicates = 3 pts penalty
        duplicate_penalty = min(duplicate_pct * 3, 30)
        # incorrect dtype penalty (up to 30 pts): 10 pts per incorrect dtype column
        incorrect_dtype_penalty = min(len(incorrect_dtypes) * 10, 30)
        
        quality_score = max(100.0 - (missing_penalty + duplicate_penalty + incorrect_dtype_penalty), 0.0)
        
        if quality_score >= 85:
            grade = "Excellent"
        elif quality_score >= 70:
            grade = "Good"
        elif quality_score >= 50:
            grade = "Fair"
        else:
            grade = "Poor"

        return {
            "duplicate_count": duplicate_count,
            "duplicate_percentage": round(duplicate_pct, 2),
            "missing_info": missing_info,
            "total_missing": total_missing,
            "total_missing_percentage": round(total_missing_pct, 2),
            "incorrect_dtypes": incorrect_dtypes,
            "quality_score": round(quality_score, 1),
            "quality_grade": grade
        }
