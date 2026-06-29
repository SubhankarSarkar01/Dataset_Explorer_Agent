import streamlit as st
import pandas as pd
from typing import Dict, Any

def inject_custom_css():
    """
    Injects custom CSS to style the Streamlit dashboard.
    Adds a premium dark/light layout, clean cards, modern typography, 
    and micro-interaction borders.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Font Overrides */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Gradient Main Title */
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            text-align: center;
            letter-spacing: -0.02em;
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: #64748B;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Styled Dashboard Cards */
        .dashboard-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .dashboard-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        }
        
        /* Metric Box Container */
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .metric-card {
            flex: 1;
            min-width: 180px;
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            color: #F8FAFC;
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
            text-align: center;
        }
        
        .metric-card-val {
            font-size: 2.2rem;
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 0.2rem;
        }
        
        .metric-card-lbl {
            font-size: 0.85rem;
            color: #94A3B8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Status Pills */
        .pill-excellent {
            background-color: #D1FAE5;
            color: #065F46;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        .pill-good {
            background-color: #DBEAFE;
            color: #1E40AF;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        .pill-fair {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        .pill-poor {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        /* Recommendation Cards */
        .rec-card {
            border-left: 5px solid #CBD5E1;
            background-color: #F8FAFC;
            border-radius: 0 12px 12px 0;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .rec-card.active {
            border-left: 5px solid #10B981;
            background-color: #ECFDF5;
        }
        
        .rec-card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1E293B;
            margin-bottom: 0.4rem;
        }
        
        .rec-card-badge {
            font-size: 0.75rem;
            padding: 0.15rem 0.5rem;
            border-radius: 9999px;
            font-weight: 600;
            text-transform: uppercase;
            margin-left: 0.5rem;
            display: inline-block;
        }
        
        .rec-card-badge.active {
            background-color: #D1FAE5;
            color: #065F46;
        }
        
        .rec-card-badge.inactive {
            background-color: #E2E8F0;
            color: #475569;
        }

        .rec-card-scope {
            font-size: 0.85rem;
            font-weight: 500;
            color: #64748B;
            margin-bottom: 0.5rem;
        }
        
        .rec-card-reason {
            font-size: 0.95rem;
            color: #334155;
            line-height: 1.5;
        }

        /* Model Recommendation Cards */
        .model-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        .model-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #F1F5F9;
            padding-bottom: 0.75rem;
            margin-bottom: 0.75rem;
        }
        
        .model-card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #4F46E5;
        }
        
        .model-card-type {
            background-color: #EEF2F6;
            color: #475569;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }
        
        </style>
    """, unsafe_allow_html=True)
