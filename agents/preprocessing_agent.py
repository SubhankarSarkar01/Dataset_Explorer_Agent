import pandas as pd
import numpy as np
from typing import Dict, Any, List

class PreprocessingAgent:
    """
    Agent responsible for generating preprocessing recommendations based on data profile:
    - Duplicates removal
    - Imputation: Median for numeric, Mode for categorical
    - Label encoding for categorical columns
    - Scaling (StandardScaler) for numeric columns
    - Drop constant columns
    """
    def __init__(self):
        pass

    def recommend(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyzes the DataFrame and returns a list of preprocessing recommendations.
        Each recommendation contains:
        - scope: The target of the recommendation (e.g. columns, global)
        - action: The recommended preprocessing operation
        - recommended: Boolean indicating if it's active or not
        - reason: Explanation of the recommendation and why it's made
        """
        recommendations = []
        
        # 1. Duplicates
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            recommendations.append({
                "scope": "Global Rows",
                "action": "Remove duplicates",
                "recommended": True,
                "reason": f"The dataset contains {dup_count} duplicate rows. Removing duplicate rows prevents training models on repeated observations, which can cause overfitting, and ensures honest performance evaluations."
            })
        else:
            recommendations.append({
                "scope": "Global Rows",
                "action": "Remove duplicates",
                "recommended": False,
                "reason": "No duplicate rows detected in the dataset. No action is required."
            })

        # 2. Drop constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)
                
        if len(constant_cols) > 0:
            col_list = ", ".join([f"'{c}'" for c in constant_cols])
            recommendations.append({
                "scope": f"Columns: {col_list}",
                "action": "Drop constant columns",
                "recommended": True,
                "reason": f"The column(s) {col_list} contain only one unique value (excluding missing values). Constant columns carry zero variance and zero predictive power. Dropping them saves memory and prevents collinearity issues."
            })
        else:
            recommendations.append({
                "scope": "All Columns",
                "action": "Drop constant columns",
                "recommended": False,
                "reason": "No constant columns detected. Every column contains multiple unique values."
            })

        # 3. Numeric Imputation (Median)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_numeric = []
        for col in numeric_cols:
            missing_cnt = int(df[col].isnull().sum())
            if missing_cnt > 0:
                missing_numeric.append((col, missing_cnt))
                
        if len(missing_numeric) > 0:
            col_details = ", ".join([f"'{c}' ({n} missing)" for c, n in missing_numeric])
            recommendations.append({
                "scope": f"Numeric Columns: {', '.join([c for c, _ in missing_numeric])}",
                "action": "Fill missing values using Median",
                "recommended": True,
                "reason": f"Missing values detected in numeric columns: {col_details}. Imputing missing values with the median is robust to outliers, protecting the central tendency of the feature without biasing models with artificial extreme values."
            })
        else:
            recommendations.append({
                "scope": "Numeric Columns",
                "action": "Fill missing values using Median",
                "recommended": False,
                "reason": "No missing values found in numeric columns. Imputation is not needed."
            })

        # 4. Categorical Imputation (Mode)
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        missing_categorical = []
        for col in categorical_cols:
            missing_cnt = int(df[col].isnull().sum())
            if missing_cnt > 0:
                missing_categorical.append((col, missing_cnt))
                
        if len(missing_categorical) > 0:
            col_details = ", ".join([f"'{c}' ({n} missing)" for c, n in missing_categorical])
            recommendations.append({
                "scope": f"Categorical Columns: {', '.join([c for c, _ in missing_categorical])}",
                "action": "Fill missing values using Mode",
                "recommended": True,
                "reason": f"Missing values detected in categorical columns: {col_details}. Imputing missing values with the mode (most frequent category) is standard practice for categorical data since mathematical averages (mean, median) cannot be calculated."
            })
        else:
            recommendations.append({
                "scope": "Categorical Columns",
                "action": "Fill missing values using Mode",
                "recommended": False,
                "reason": "No missing values found in categorical columns. Imputation is not needed."
            })

        # 5. Label Encode Categorical Columns
        if len(categorical_cols) > 0:
            col_list = ", ".join([f"'{c}'" for c in categorical_cols])
            recommendations.append({
                "scope": f"Categorical Columns: {col_list}",
                "action": "Label encode categorical columns",
                "recommended": True,
                "reason": f"Categorical columns detected: {col_list}. Scikit-learn machine learning algorithms require numerical input matrices. Label encoding (or mapping category strings to numeric integers) converts text categories into a form the models can process."
            })
        else:
            recommendations.append({
                "scope": "Categorical Columns",
                "action": "Label encode categorical columns",
                "recommended": False,
                "reason": "No categorical columns found in the dataset. Encoding is not needed."
            })

        # 6. StandardScaler for Numeric Features
        if len(numeric_cols) > 0:
            stds = df[numeric_cols].std()
            max_std = stds.max()
            min_std = stds.min()
            
            scale_note = ""
            if not pd.isna(max_std) and not pd.isna(min_std) and min_std > 0:
                ratio = max_std / min_std
                if ratio > 10.0:
                    scale_note = f" (Note: Feature scales differ drastically, with standard deviations ranging from {min_std:.2f} to {max_std:.2f})."
            
            recommendations.append({
                "scope": f"Numeric Features: {', '.join([f'\'{c}\'' for c in numeric_cols])}",
                "action": "StandardScaler scaling",
                "recommended": True,
                "reason": f"Numeric columns exist in the dataset{scale_note}. Standardizing numeric columns to unit variance and zero mean is essential for scale-sensitive algorithms (like Logistic Regression, SVM, and k-NN) to prevent larger scale values from disproportionately dominating optimization and distances."
            })
        else:
            recommendations.append({
                "scope": "Numeric Features",
                "action": "StandardScaler scaling",
                "recommended": False,
                "reason": "No numeric features found to scale."
            })

        return recommendations
