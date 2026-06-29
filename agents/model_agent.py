import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class ModelAgent:
    """
    Agent responsible for detecting the ML problem type (Classification vs Regression)
    and recommending models with detailed justifications.
    """
    def __init__(self):
        pass

    def detect_problem_type(self, df: pd.DataFrame, target_col: str) -> str:
        """
        Detects the ML problem type based on the target column's dtype and unique values.
        """
        series = df[target_col].dropna()
        if series.empty:
            return "Unknown"
            
        dtype = series.dtype
        unique_count = series.nunique()
        
        # 1. String, Category, Boolean types -> Classification
        if dtype == 'object' or isinstance(dtype, pd.CategoricalDtype) or dtype == 'bool':
            return "Classification"
            
        # 2. Float type -> Regression
        if np.issubdtype(dtype, np.floating):
            return "Regression"
            
        # 3. Integer type -> Cardinarity heuristic
        if np.issubdtype(dtype, np.integer):
            # If the column has few unique values (e.g. <= 15), it is likely representing classes/labels
            if unique_count <= 15:
                return "Classification"
            else:
                return "Regression"
                
        return "Classification" # Fallback

    def recommend(self, df: pd.DataFrame, target_col: Optional[str]) -> Dict[str, Any]:
        """
        Generates ML model recommendations based on the target column selection.
        """
        if not target_col or target_col not in df.columns:
            return {
                "target_column": None,
                "problem_detected": "N/A",
                "recommended": False,
                "models": [],
                "explanation": "No target column has been selected. Supervised machine learning model recommendations cannot be made without a designated target (label) column. Please select a target column in the sidebar."
            }
            
        problem_type = self.detect_problem_type(df, target_col)
        unique_vals = int(df[target_col].nunique())
        
        models = []
        if problem_type == "Classification":
            models = [
                {
                    "name": "Logistic Regression",
                    "type": "Classification",
                    "pros": "Highly interpretable, fast training and inference, outputs probability estimates, easily regularized (L1/L2).",
                    "cons": "Assumes linear boundary; struggles with complex non-linear interactions without manual feature engineering.",
                    "reason": "Excellent baseline. Use this to determine linear separability of classes and when decision transparency is required."
                },
                {
                    "name": "Random Forest Classifier",
                    "type": "Classification",
                    "pros": "Strong out-of-the-box accuracy, handles non-linear relationships, robust to outliers and noise, provides feature importances.",
                    "cons": "Can be memory-heavy, slow to predict on massive datasets, functions as a 'black-box' ensemble.",
                    "reason": "Usually the top-performing non-linear baseline. Very robust for tabular datasets, requiring minimal hyperparameter tuning."
                },
                {
                    "name": "Decision Tree Classifier",
                    "type": "Classification",
                    "pros": "Extremely interpretable (visualizable structure), handles mixed data types natively, requires zero feature scaling.",
                    "cons": "Highly prone to overfitting (high variance) if tree depth is not restricted, highly sensitive to training data perturbations.",
                    "reason": "Good for generating simple, clear decision rules and understanding hierarchical feature relationships."
                },
                {
                    "name": "Support Vector Machine (SVM)",
                    "type": "Classification",
                    "pros": "Effective in high-dimensional spaces, versatile kernel tricks (RBF, Polynomial) allow modeling of complex boundaries, robust against overfitting.",
                    "cons": "Requires feature scaling (e.g. StandardScaler), computationally expensive for large datasets (O(n_samples^2 * n_features)), lacks direct probability mapping.",
                    "reason": "Strong choice if classes are separated by a complex non-linear boundary, especially in medium-sized datasets. Must scale features first."
                }
            ]
        elif problem_type == "Regression":
            models = [
                {
                    "name": "Linear Regression",
                    "type": "Regression",
                    "pros": "Fast to compute, highly interpretable coefficients, serves as a standard reference point.",
                    "cons": "Assumes strict linearity, highly sensitive to multicollinearity and outlier values.",
                    "reason": "Fundamental baseline model. Tells you if a simple linear combination of inputs can successfully approximate the target."
                },
                {
                    "name": "Random Forest Regressor",
                    "type": "Regression",
                    "pros": "Excellent accuracy, models complex non-linear relationships, handles high dimensional data, immune to outliers.",
                    "cons": "Cannot extrapolate outside the training target range (e.g. cannot predict a value higher than the training maximum), complex ensemble.",
                    "reason": "Strongest non-linear regression baseline. Highly recommended for standard tabular datasets where relationships are non-linear."
                },
                {
                    "name": "Decision Tree Regressor",
                    "type": "Regression",
                    "pros": "Interpretable, requires no scaling, captures non-linear trends.",
                    "cons": "Unstable predictions, prone to overfitting, creates discontinuous step-like regression predictions.",
                    "reason": "Helpful baseline model to explore tree structures and rule-based splits before using forest ensembles."
                }
            ]
            
        return {
            "target_column": target_col,
            "problem_detected": problem_type,
            "recommended": True,
            "unique_values": unique_vals,
            "models": models,
            "explanation": f"Based on target column '{target_col}' (type: {df[target_col].dtype}, unique values: {unique_vals}), the task was classified as a **{problem_type}** problem. Below are the recommended models:"
        }
