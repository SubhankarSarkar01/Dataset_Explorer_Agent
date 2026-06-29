import os
from agents.loader_agent import LoaderAgent
from agents.quality_agent import DataQualityAgent
from agents.eda_agent import EDAAgent
from agents.outlier_agent import OutlierAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_agent import ModelAgent
from agents.report_agent import ReportAgent

def main():
    print("=== TESTING DATASET EXPLORER AGENT OFFLINE PIPELINE ===")
    
    # 1. Loader Agent
    print("\nRunning LoaderAgent...")
    loader = LoaderAgent()
    df = loader.load_data("sample_data.csv")
    summary = loader.get_summary(df)
    print(f"Shape loaded: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    
    # 2. Quality Agent
    print("\nRunning DataQualityAgent...")
    quality = DataQualityAgent()
    q_res = quality.analyze(df)
    print(f"Score: {q_res['quality_score']}/100 ({q_res['quality_grade']})")
    print(f"Duplicates: {q_res['duplicate_count']}")
    print(f"Total Missing: {q_res['total_missing']}")
    print(f"Incorrect Dtypes detected: {q_res['incorrect_dtypes']}")
    
    # 3. EDA Agent
    print("\nRunning EDAAgent...")
    eda = EDAAgent()
    stats = eda.get_descriptive_stats(df)
    print("Numerical stats columns:", list(stats['numeric'].columns))
    print("Categorical stats columns:", list(stats['categorical'].columns))
    
    # 4. Outlier Agent
    print("\nRunning OutlierAgent...")
    outliers = OutlierAgent()
    o_res = outliers.analyze_outliers(df)
    print(f"Total IQR Outliers: {o_res['total_iqr_outliers']}")
    print(f"Total Z-Score Outliers: {o_res['total_zscore_outliers']}")
    
    # 5. Preprocessing Agent
    print("\nRunning PreprocessingAgent...")
    prep = PreprocessingAgent()
    recs = prep.recommend(df)
    for r in recs:
        print(f"- Action: {r['action']}, Recommended: {r['recommended']}, Scope: {r['scope']}")
        
    # 6. Model Agent
    print("\nRunning ModelAgent...")
    model = ModelAgent()
    # Test with target selected (classification)
    m_res = model.recommend(df, "purchased")
    print(f"Target 'purchased' detected problem type: {m_res['problem_detected']}")
    print("Recommended models:")
    for m in m_res['models']:
        print(f"  * {m['name']}")
        
    # Test with target selected (regression, age column)
    m_res_reg = model.recommend(df, "age")
    print(f"Target 'age' detected problem type: {m_res_reg['problem_detected']}")
    print("Recommended models:")
    for m in m_res_reg['models']:
        print(f"  * {m['name']}")
        
    # Test with no target
    m_res_none = model.recommend(df, None)
    print(f"Target 'None' recommended flag: {m_res_none['recommended']}")
    
    # 7. Report Agent
    print("\nRunning ReportAgent...")
    reporter = ReportAgent()
    report_md = reporter.generate_report(summary, q_res, o_res, recs, m_res)
    
    print("\nSaving Report to 'reports/sample_report.md'...")
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "sample_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
    print("Sample report generated successfully.")

if __name__ == "__main__":
    main()
