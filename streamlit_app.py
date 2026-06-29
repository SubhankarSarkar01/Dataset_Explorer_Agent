import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page configurations (tab title, layout, etc.)
st.set_page_config(
    page_title="Dataset Explorer Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our custom agents and helpers
from agents.loader_agent import LoaderAgent
from agents.quality_agent import DataQualityAgent
from agents.eda_agent import EDAAgent
from agents.outlier_agent import OutlierAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_agent import ModelAgent
from agents.report_agent import ReportAgent
from utils.helpers import inject_custom_css

# Inject modern, premium aesthetic CSS overrides
inject_custom_css()

# Instantiate agents
loader_agent = LoaderAgent()
quality_agent = DataQualityAgent()
eda_agent = EDAAgent()
outlier_agent = OutlierAgent()
preprocess_agent = PreprocessingAgent()
model_agent = ModelAgent()
report_agent = ReportAgent()

# Initialize session state variables to track inputs and analysis results
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'loader_summary' not in st.session_state:
    st.session_state.loader_summary = None
if 'quality_results' not in st.session_state:
    st.session_state.quality_results = None
if 'outlier_results' not in st.session_state:
    st.session_state.outlier_results = None
if 'preprocessing_recs' not in st.session_state:
    st.session_state.preprocessing_recs = None
if 'model_recs' not in st.session_state:
    st.session_state.model_recs = None
if 'markdown_report' not in st.session_state:
    st.session_state.markdown_report = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""

# --- Sidebar Controls ---
st.sidebar.markdown("## ⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# Reset or Load dataset if file selection changes
if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        try:
            # Loader Agent executes first action
            df = loader_agent.load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.loader_summary = loader_agent.get_summary(df)
            st.session_state.file_name = uploaded_file.name
            st.session_state.analysis_run = False  # Reset analysis state for new file
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            st.session_state.df = None
else:
    st.session_state.df = None
    st.session_state.loader_summary = None
    st.session_state.file_name = ""
    st.session_state.analysis_run = False

# Target Column selector (optional)
target_col = None
if st.session_state.df is not None:
    cols = ["None"] + list(st.session_state.df.columns)
    selected_target = st.sidebar.selectbox("Select Target Column (Optional)", cols)
    if selected_target != "None":
        target_col = selected_target
    st.session_state.target_column = target_col

# Trigger sequential pipeline execution
if st.session_state.df is not None:
    run_btn = st.sidebar.button("Run Complete Analysis", type="primary", use_container_width=True)
    if run_btn:
        with st.spinner("Executing sequence of specialized agents..."):
            df = st.session_state.df
            
            # 1. Data Quality Agent
            quality_res = quality_agent.analyze(df)
            st.session_state.quality_results = quality_res
            
            # 2. Outlier Agent
            outlier_res = outlier_agent.analyze_outliers(df)
            st.session_state.outlier_results = outlier_res
            
            # 3. Preprocessing Recommendation Agent
            prep_recs = preprocess_agent.recommend(df)
            st.session_state.preprocessing_recs = prep_recs
            
            # 4. Model Recommendation Agent
            model_recs = model_agent.recommend(df, st.session_state.target_column)
            st.session_state.model_recs = model_recs
            
            # 5. Report Agent
            report_md = report_agent.generate_report(
                st.session_state.loader_summary,
                quality_res,
                outlier_res,
                prep_recs,
                model_recs
            )
            st.session_state.markdown_report = report_md
            st.session_state.analysis_run = True
            st.toast("Analysis Completed!", icon="🚀")
else:
    st.sidebar.info("Upload a CSV file in the sidebar to begin analysis.")

# --- Main Dashboard Content ---
st.markdown('<div class="main-title">Dataset Explorer Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An offline multi-agent system for automated data profiling, data quality grading, exploratory analysis, and model recommendations.</div>', unsafe_allow_html=True)

if st.session_state.df is not None:
    if st.session_state.analysis_run:
        # Create Tabs for visual layout
        tab_preview, tab_quality, tab_eda, tab_outliers, tab_recs, tab_report = st.tabs([
            "📊 Dataset Preview",
            "🛡️ Data Quality",
            "📈 Exploratory Analysis (EDA)",
            "🎯 Outliers Detection",
            "💡 Recommendations",
            "📄 Generated Report"
        ])
        
        # --- TAB 1: DATASET PREVIEW ---
        with tab_preview:
            st.markdown("### 📋 Dataset Overview")
            
            # Custom styled metric blocks
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-card-val">{st.session_state.loader_summary['rows']:,}</div>
                        <div class="metric-card-lbl">Total Rows</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val">{st.session_state.loader_summary['num_columns']:,}</div>
                        <div class="metric-card-lbl">Total Columns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val">{st.session_state.df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB</div>
                        <div class="metric-card-lbl">Memory Footprint</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Sample Preview (First 5 Rows)")
            st.dataframe(st.session_state.df.head(5), use_container_width=True)
            
            st.markdown("#### Column Metadata Summary")
            meta_rows = []
            for col in st.session_state.df.columns:
                meta_rows.append({
                    "Column": col,
                    "Data Type": str(st.session_state.df[col].dtype),
                    "Unique Values": st.session_state.df[col].nunique(),
                    "Non-Null Count": st.session_state.df[col].count()
                })
            st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)
            
        # --- TAB 2: DATA QUALITY ---
        with tab_quality:
            st.markdown("### 🛡️ Data Quality Analysis")
            qr = st.session_state.quality_results
            
            # Choose correct CSS class based on grade
            grade_class = "pill-poor"
            if qr['quality_grade'] == "Excellent":
                grade_class = "pill-excellent"
            elif qr['quality_grade'] == "Good":
                grade_class = "pill-good"
            elif qr['quality_grade'] == "Fair":
                grade_class = "pill-fair"
                
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-card-val">{qr['quality_score']}/100</div>
                        <div class="metric-card-lbl">Quality Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val"><span class="{grade_class}">{qr['quality_grade']}</span></div>
                        <div class="metric-card-lbl">Quality Grade</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val">{qr['duplicate_count']:,}</div>
                        <div class="metric-card-lbl">Duplicate Rows</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val">{qr['total_missing']:,} ({qr['total_missing_percentage']}%)</div>
                        <div class="metric-card-lbl">Missing Cells</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Type warnings
            if qr['incorrect_dtypes']:
                st.warning("⚠️ **Potential Schema/Data Type Inconsistencies Detected:**")
                for dtype_issue in qr['incorrect_dtypes']:
                    st.markdown(f"""
                        <div class="dashboard-card">
                            <span style="font-weight:600;color:#DC2626;font-size:1.05rem;">Column: {dtype_issue['column']}</span><br>
                            <span style="font-size:0.9rem;color:#64748B;">• Current Data Type: <code>{dtype_issue['current_type']}</code> | Suggested: <code>{dtype_issue['suggested_type']}</code></span><br>
                            <div style="margin-top:0.4rem;font-size:0.95rem;"><strong>Justification:</strong> {dtype_issue['reason']}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No schema / incorrect data type inconsistencies detected in object columns.")
                
            st.markdown("#### Missing Values Count and Percentage by Column")
            missing_df = pd.DataFrame(qr['missing_info'])
            st.dataframe(missing_df.rename(columns={
                "column": "Column Name",
                "missing_count": "Missing Rows",
                "missing_percentage": "Missing Percentage (%)",
                "dtype": "Data Type"
            }), use_container_width=True)

        # --- TAB 3: EXPLORATORY ANALYSIS (EDA) ---
        with tab_eda:
            st.markdown("### 📈 Descriptive Statistics and Visualization")
            
            eda_stats = eda_agent.get_descriptive_stats(st.session_state.df)
            
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.markdown("#### Numerical Columns Statistics")
                if not eda_stats["numeric"].empty:
                    st.dataframe(eda_stats["numeric"], use_container_width=True)
                else:
                    st.info("No numeric columns available in the dataset.")
            with col_stats2:
                st.markdown("#### Categorical Columns Statistics")
                if not eda_stats["categorical"].empty:
                    st.dataframe(eda_stats["categorical"], use_container_width=True)
                else:
                    st.info("No categorical/object columns available in the dataset.")

            # Correlation Heatmap
            st.markdown("#### Pearson Correlation Heatmap")
            corr_fig = eda_agent.plot_correlation_heatmap(st.session_state.df)
            if corr_fig is not None:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("Correlation analysis requires at least 2 numerical columns.")
                
            # Column-level plotting
            st.markdown("#### Interactive Distribution & Boxplot Visualizations")
            numeric_cols = list(st.session_state.df.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols) > 0:
                selected_num_col = st.selectbox("Select Numeric Column to Plot", numeric_cols)
                
                col_plot1, col_plot2 = st.columns(2)
                with col_plot1:
                    hist_fig = eda_agent.plot_histogram(st.session_state.df, selected_num_col)
                    if hist_fig:
                        st.plotly_chart(hist_fig, use_container_width=True)
                with col_plot2:
                    box_fig = eda_agent.plot_boxplot(st.session_state.df, selected_num_col)
                    if box_fig:
                        st.plotly_chart(box_fig, use_container_width=True)
            else:
                st.info("No numerical columns found to plot distributions.")
                
            # Categorical Value Counts
            st.markdown("#### Interactive Categorical Value Counts")
            categorical_cols = list(st.session_state.df.select_dtypes(include=['object', 'category', 'bool']).columns)
            if len(categorical_cols) > 0:
                selected_cat_col = st.selectbox("Select Categorical Column to Analyze", categorical_cols)
                cat_counts = eda_agent.get_value_counts(st.session_state.df, selected_cat_col)
                if not cat_counts.empty:
                    st.dataframe(cat_counts, use_container_width=True)
            else:
                st.info("No categorical columns found in dataset.")

        # --- TAB 4: OUTLIERS ---
        with tab_outliers:
            st.markdown("### 🎯 Outlier Analysis Summary")
            out_res = st.session_state.outlier_results
            
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-card-val">{out_res['total_iqr_outliers']:,}</div>
                        <div class="metric-card-lbl">Total IQR Outliers</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val">{out_res['total_zscore_outliers']:,}</div>
                        <div class="metric-card-lbl">Total Z-Score Outliers (|Z| > 3)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val">{out_res['num_numeric_columns']}</div>
                        <div class="metric-card-lbl">Numeric Columns Scanned</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if out_res['columns']:
                st.markdown("#### Numerical Column Outlier Metrics")
                out_summary_list = []
                for col, summary in out_res['columns'].items():
                    out_summary_list.append({
                        "Column": col,
                        "Mean": f"{summary['mean']:.3f}",
                        "Std Dev": f"{summary['std']:.3f}",
                        "IQR Outlier Count": summary['iqr_count'],
                        "IQR Lower Boundary": f"{summary['iqr_bounds'][0]:.2f}",
                        "IQR Upper Boundary": f"{summary['iqr_bounds'][1]:.2f}",
                        "Z-Score Outlier Count": summary['zscore_count']
                    })
                st.dataframe(pd.DataFrame(out_summary_list), use_container_width=True)
            else:
                st.info("No numeric columns available for outlier detection.")

        # --- TAB 5: RECOMMENDATIONS ---
        with tab_recs:
            st.markdown("### 💡 Automated Pipeline Decisions")
            
            # Preprocessing
            st.markdown("#### Preprocessing Recommendations")
            for rec in st.session_state.preprocessing_recs:
                card_class = "rec-card active" if rec['recommended'] else "rec-card"
                badge_class = "rec-card-badge active" if rec['recommended'] else "rec-card-badge inactive"
                badge_txt = "Active Recommendation" if rec['recommended'] else "Not Needed"
                
                st.markdown(f"""
                    <div class="{card_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="rec-card-title">⚙️ {rec['action']}</span>
                            <span class="{badge_class}">{badge_txt}</span>
                        </div>
                        <div class="rec-card-scope">🎯 Scope: <code>{rec['scope']}</code></div>
                        <div class="rec-card-reason">{rec['reason']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            # Machine Learning Modeling
            st.markdown("#### Machine Learning Model Recommendations")
            mr = st.session_state.model_recs
            
            st.info(mr['explanation'])
            
            if mr['recommended'] and len(mr['models']) > 0:
                for model in mr['models']:
                    st.markdown(f"""
                        <div class="model-card">
                            <div class="model-card-header">
                                <span class="model-card-title">🤖 {model['name']}</span>
                                <span class="model-card-type">{model['type']}</span>
                            </div>
                            <p><strong>Rationale:</strong> {model['reason']}</p>
                            <p>🟢 <strong>Pros:</strong> {model['pros']}</p>
                            <p>🔴 <strong>Cons:</strong> {model['cons']}</p>
                        </div>
                    """, unsafe_allow_html=True)

        # --- TAB 6: GENERATED REPORT ---
        with tab_report:
            st.markdown("### 📄 Generated Markdown Report")
            
            # Downloader widget
            report_name = f"dataset_report_{os.path.splitext(st.session_state.file_name)[0]}.md"
            st.download_button(
                label="📥 Download Markdown Report",
                data=st.session_state.markdown_report,
                file_name=report_name,
                mime="text/markdown",
                use_container_width=True
            )
            
            st.markdown("---")
            # Display report content as markdown block
            st.markdown(st.session_state.markdown_report)
            
    else:
        # File loaded, waiting to execute
        st.info("👈 Dataset loaded successfully! Review the quick preview below, choose a Target Column in the sidebar if desired, and click **'Run Complete Analysis'** to start the pipeline.")
        
        st.markdown("### 📋 Quick Preview")
        st.dataframe(st.session_state.df.head(5), use_container_width=True)
else:
    # Empty placeholder Landing Page
    st.markdown("""
        <div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 16px; padding: 4rem 2rem; text-align: center; margin-top: 2rem;">
            <h3 style="color:#1E293B; margin-bottom: 1rem; font-size:1.6rem; font-weight:600;">Welcome to the Dataset Explorer Agent!</h3>
            <p style="color:#64748B; font-size:1.1rem; max-width: 600px; margin: 0 auto 2.5rem auto; line-height:1.6;">
                Analyze your CSV data completely offline. Upload your dataset using the sidebar file uploader. We support automatic quality grading, distribution profiling, outlier detection, preprocessing steps, and machine learning models selection.
            </p>
            <div style="color: #4F46E5; font-weight:600; font-size:0.95rem; letter-spacing:0.02em;">
                📂 LOADER AGENT &nbsp;•&nbsp; 🛡️ DATA QUALITY AGENT &nbsp;•&nbsp; 📈 EDA AGENT &nbsp;•&nbsp; 🎯 OUTLIER AGENT &nbsp;•&nbsp; 💡 PIPELINE RECS
            </div>
        </div>
    """, unsafe_allow_html=True)
