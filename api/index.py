import sys
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import plotly.utils

# Add the parent directory of 'api' to the python path to allow importing agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.loader_agent import LoaderAgent
from agents.quality_agent import DataQualityAgent
from agents.eda_agent import EDAAgent
from agents.outlier_agent import OutlierAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_agent import ModelAgent
from agents.report_agent import ReportAgent

app = Flask(__name__)
# Enable CORS for easier testing/development
CORS(app)

# Instantiate agents
loader_agent = LoaderAgent()
quality_agent = DataQualityAgent()
eda_agent = EDAAgent()
outlier_agent = OutlierAgent()
preprocess_agent = PreprocessingAgent()
model_agent = ModelAgent()
report_agent = ReportAgent()

@app.route('/api/preview', methods=['POST'])
def preview():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    try:
        df = loader_agent.load_data(file)
        summary = loader_agent.get_summary(df)
        
        preview_records = summary['preview'].to_dict(orient="records")
        # Ensure NaNs are converted to None for valid JSON serialization
        preview_records = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in preview_records]
        
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        column_metadata = []
        for col in df.columns:
            column_metadata.append({
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Unique Values": int(df[col].nunique()),
                "Non-Null Count": int(df[col].count())
            })
            
        return jsonify({
            "rows": int(summary['rows']),
            "columns": summary['columns'],
            "num_columns": int(summary['num_columns']),
            "preview": preview_records,
            "memory_usage": round(float(memory_usage), 2),
            "column_metadata": column_metadata
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    target_column = request.form.get('target_column', None)
    if target_column == "None" or target_column == "":
        target_column = None
        
    try:
        # 1. Load Data
        df = loader_agent.load_data(file)
        
        # 2. Get Loader Summary
        loader_summary = loader_agent.get_summary(df)
        preview_records = loader_summary['preview'].to_dict(orient="records")
        preview_records = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in preview_records]
        
        loader_summary_json = {
            "rows": int(loader_summary['rows']),
            "columns": loader_summary['columns'],
            "num_columns": int(loader_summary['num_columns']),
            "preview": preview_records,
            "memory_usage": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2)
        }
        
        # 3. Data Quality
        quality_res = quality_agent.analyze(df)
        # Convert missing info dict/NaNs to float/int/None safe values
        quality_res["missing_info"] = [
            {
                "column": x["column"],
                "missing_count": int(x["missing_count"]),
                "missing_percentage": float(x["missing_percentage"]),
                "dtype": str(x["dtype"])
            }
            for x in quality_res["missing_info"]
        ]
        
        # 4. Outliers
        outlier_res = outlier_agent.analyze_outliers(df)
        
        # 5. Preprocessing
        prep_recs = preprocess_agent.recommend(df)
        
        # 6. Model Recommendations
        model_recs = model_agent.recommend(df, target_column)
        
        # 7. EDA
        eda_stats = eda_agent.get_descriptive_stats(df)
        
        # Convert descriptive stats DataFrames to serializable dicts
        # pd.DataFrame.to_dict(orient="split") or "index" is great
        numeric_stats = {}
        if not eda_stats["numeric"].empty:
            # Transpose descriptive stats so columns are metrics (count, mean, std...)
            numeric_stats_df = eda_stats["numeric"].reset_index().rename(columns={"index": "Metric"})
            numeric_stats_records = numeric_stats_df.to_dict(orient="records")
            numeric_stats = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in numeric_stats_records]
            
        categorical_stats = {}
        if not eda_stats["categorical"].empty:
            categorical_stats_df = eda_stats["categorical"].reset_index().rename(columns={"index": "Metric"})
            categorical_stats_records = categorical_stats_df.to_dict(orient="records")
            categorical_stats = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in categorical_stats_records]
            
        # Pearson Correlation Heatmap
        corr_fig = eda_agent.plot_correlation_heatmap(df)
        corr_heatmap_json = json.loads(json.dumps(corr_fig, cls=plotly.utils.PlotlyJSONEncoder)) if corr_fig else None
        
        # Pre-generate distributions (histograms & box plots) for all numeric columns
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        eda_plots = {}
        for col in numeric_cols:
            hist_fig = eda_agent.plot_histogram(df, col)
            box_fig = eda_agent.plot_boxplot(df, col)
            eda_plots[col] = {
                "histogram": json.loads(json.dumps(hist_fig, cls=plotly.utils.PlotlyJSONEncoder)) if hist_fig else None,
                "boxplot": json.loads(json.dumps(box_fig, cls=plotly.utils.PlotlyJSONEncoder)) if box_fig else None
            }
            
        # Pre-generate categorical value counts
        categorical_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)
        cat_counts = {}
        for col in categorical_cols:
            val_counts_df = eda_agent.get_value_counts(df, col)
            cat_counts[col] = val_counts_df.to_dict(orient="records") if not val_counts_df.empty else []
            
        # 8. Report Generation
        report_md = report_agent.generate_report(
            loader_summary,
            quality_res,
            outlier_res,
            prep_recs,
            model_recs
        )
        
        response_data = {
            "loader_summary": loader_summary_json,
            "quality_results": quality_res,
            "outlier_results": outlier_res,
            "preprocessing_recs": prep_recs,
            "model_recs": model_recs,
            "eda": {
                "numeric_stats": numeric_stats,
                "categorical_stats": categorical_stats,
                "correlation_heatmap": corr_heatmap_json,
                "plots": eda_plots,
                "cat_counts": cat_counts,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols
            },
            "markdown_report": report_md
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# For local running/debugging
if __name__ == '__main__':
    app.run(port=5000, debug=True)
