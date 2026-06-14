# =============================================================================
# RT-DeepNIDS — app.py  (Frontend / Streamlit entry point)
# A Real-Time Hybrid Network Intrusion Detection System
# Bangladesh University of Business and Technology (BUBT)
# Department of Computer Science and Engineering
#
# Run:  streamlit run app.py
# =============================================================================


import os
import html as html_mod
from datetime import datetime


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# All ML / feature-engineering logic lives in backend.py
from engine import (
   DL_MODELS, DATASET_COLS,
   EXCLUDE_COLS, MODEL_FILE_MAP,
   load_framework_assets, load_csv_source,
   get_feature_count, clean_label, safe_int,
   resolve_dataset_dir, resolve_model_file,
)




# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
   page_title="RT-DeepNIDS",
   layout="wide",
   initial_sidebar_state="expanded",
)




# =============================================================================
# THEME SYSTEM
# =============================================================================
if "theme" not in st.session_state:
   st.session_state.theme = "light"


T_DARK = {
   "page_bg":         "#050810",
   "sidebar_bg":      "linear-gradient(180deg,#080d1a 0%,#050810 100%)",
   "sidebar_border":  "#1e3a5f",
   "card_bg":         "linear-gradient(145deg,#0f1729 0%,#0a1020 60%,#060c18 100%)",
   "card_border":     "#1e3a5f",
   "card_border_t":   "#2a4a7f",
   "card_shadow":     "0 8px 32px rgba(0,0,0,0.6),0 2px 8px rgba(14,165,233,0.08),inset 0 1px 0 rgba(255,255,255,0.04)",
   "label_color":     "#64748b",
   "value_color":     "#e2e8f0",
   "text_pri":        "#e2e8f0",
   "text_sec":        "#94a3b8",
   "text_muted":      "#475569",
   "accent":          "#38bdf8",
   "accent_dim":      "rgba(56,189,248,0.08)",
   "border":          "#1e3a5f",
   "grid":            "rgba(30,58,95,0.5)",
   "tick":            "#475569",
   "threat_box_bg":   "linear-gradient(145deg,#090e1c,#060b16)",
   "progress_track":  "#0f1a2e",
   "tab_border":      "#1e3a5f",
   "tab_color":       "#64748b",
   "input_bg":        "#0a1428",
   "input_border":    "#1e3a5f",
   "nav_bg":          "#050810",
   "nav_border":      "#1e3a5f",
   "nav_text":        "#e2e8f0",
   "btn_start_bg":    "linear-gradient(145deg,#0ea5e9,#0284c7)",
   "btn_start_color": "#ffffff",
   "btn_start_shadow":"0 4px 12px rgba(14,165,233,0.4),inset 0 1px 0 rgba(255,255,255,0.15)",
   "btn_stop_bg":     "linear-gradient(145deg,#1e293b,#0f172a)",
   "btn_stop_color":  "#94a3b8",
   "btn_stop_border": "#1e3a5f",
   "btn_reset_bg":    "#0a1428",
   "btn_reset_color": "#64748b",
   "btn_reset_border":"#1e3a5f",
   "df_bg":           "#080d1a",
   "df_border":       "#1e3a5f",
   "radio_color":     "#94a3b8",
}


T_LIGHT = {
   "page_bg":         "#f1f5f9",
   "sidebar_bg":      "linear-gradient(180deg,#ffffff 0%,#f8fafc 100%)",
   "sidebar_border":  "#cbd5e1",
   "card_bg":         "linear-gradient(145deg,#ffffff 0%,#f8fafc 60%,#f1f5f9 100%)",
   "card_border":     "#e2e8f0",
   "card_border_t":   "#cbd5e1",
   "card_shadow":     "0 4px 16px rgba(0,0,0,0.06),0 1px 4px rgba(14,165,233,0.04),inset 0 1px 0 rgba(255,255,255,0.9)",
   "label_color":     "#64748b",
   "value_color":     "#0f172a",
   "text_pri":        "#0f172a",
   "text_sec":        "#334155",
   "text_muted":      "#64748b",
   "accent":          "#0284c7",
   "accent_dim":      "rgba(2,132,199,0.08)",
   "border":          "#e2e8f0",
   "grid":            "rgba(148,163,184,0.25)",
   "tick":            "#94a3b8",
   "threat_box_bg":   "linear-gradient(145deg,#ffffff,#f8fafc)",
   "progress_track":  "#e2e8f0",
   "tab_border":      "#e2e8f0",
   "tab_color":       "#64748b",
   "input_bg":        "#ffffff",
   "input_border":    "#cbd5e1",
   "nav_bg":          "#ffffff",
   "nav_border":      "#e2e8f0",
   "nav_text":        "#0f172a",
   "btn_start_bg":    "linear-gradient(145deg,#0ea5e9,#0284c7)",
   "btn_start_color": "#ffffff",
   "btn_start_shadow":"0 4px 12px rgba(14,165,233,0.3),inset 0 1px 0 rgba(255,255,255,0.2)",
   "btn_stop_bg":     "linear-gradient(145deg,#f1f5f9,#e2e8f0)",
   "btn_stop_color":  "#475569",
   "btn_stop_border": "#cbd5e1",
   "btn_reset_bg":    "#f8fafc",
   "btn_reset_color": "#64748b",
   "btn_reset_border":"#e2e8f0",
   "df_bg":           "#ffffff",
   "df_border":       "#e2e8f0",
   "radio_color":     "#334155",
}


C = T_DARK if st.session_state.theme == "dark" else T_LIGHT




# =============================================================================
# GLOBAL CSS
# =============================================================================
def _build_css(c: dict) -> str:
   return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Exo+2:wght@300;400;600;700;800&display=swap');


/* --- SUPPRESS STREAMLIT CHROME --- */
[data-testid="stStatusWidget"]{{visibility:hidden!important;display:none!important;}}
[data-testid="stHeader"]{{display:none!important;visibility:hidden!important;height:0!important;}}
[data-testid="stToolbar"]{{display:none!important;visibility:hidden!important;height:0!important;}}
[data-testid="stDecoration"]{{display:none!important;}}
[data-testid="stMainMenuButton"]{{display:none!important;}}
header[data-testid="stHeader"]{{display:none!important;}}


/* --- PREVENT RERUN FLASH / OPACITY SHAKE --- */
div[data-stale],.stElementContainer,.stPlotlyChart,.stDataFrame,
div[data-testid="stMarkdownContainer"],
[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],.main{{
   opacity:1!important;filter:none!important;
   transition:none!important;animation:none!important;transform:none!important;
}}
div[data-stale="true"],div[data-stale="true"] *{{
   opacity:1!important;filter:none!important;
   transition:none!important;animation:none!important;
}}


/* --- GLOBAL --- */
html,body{{background-color:{c["page_bg"]}!important;overflow-x:hidden!important;}}
[data-testid="stAppViewContainer"]{{background-color:{c["page_bg"]}!important;}}
[data-testid="stAppViewContainer"] *{{
   color:{c["text_pri"]}!important;
   font-family:'Exo 2',sans-serif!important;
}}
.main .block-container{{
   padding-top:1.5rem!important;
   padding-bottom:5rem!important;
   background:transparent!important;
   max-width:100%!important;
}}


/* --- RESPONSIVE COLUMNS --- */
@media (max-width: 768px) {{
   .main .block-container{{padding-left:0.75rem!important;padding-right:0.75rem!important;}}
   [data-testid="stMetric"]{{height:auto!important;min-height:90px!important;}}
}}


/* --- KPI METRIC CARDS --- */
[data-testid="stMetric"]{{
   background:{c["card_bg"]}!important;
   border:1px solid {c["card_border"]}!important;
   border-top:2px solid {c["card_border_t"]}!important;
   border-radius:10px!important;
   padding:16px 18px!important;
   height:110px!important;
   box-shadow:{c["card_shadow"]}!important;
   transform:translateZ(0);
}}
[data-testid="stMetricLabel"]>div{{
   font-size:10px!important;font-weight:700!important;
   color:{c["label_color"]}!important;text-transform:uppercase!important;
   letter-spacing:1.2px!important;font-family:'JetBrains Mono',monospace!important;
}}
[data-testid="stMetricValue"]>div{{
   font-size:28px!important;font-weight:800!important;
   color:{c["value_color"]}!important;font-family:'Rajdhani',sans-serif!important;
   letter-spacing:1px!important;
}}


/* --- SIDEBAR --- */
[data-testid="stSidebar"]{{
   background:{c["sidebar_bg"]}!important;
   border-right:1px solid {c["sidebar_border"]}!important;
   box-shadow:4px 0 24px rgba(0,0,0,0.08)!important;
}}
[data-testid="stSidebar"] *{{color:{c["text_sec"]}!important;}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{{color:{c["text_pri"]}!important;}}
[data-testid="stSidebar"] hr{{border-color:{c["border"]}!important;opacity:0.6!important;}}


/* --- SIDEBAR RADIO --- */
[data-testid="stSidebar"] [data-testid="stRadio"] label{{color:{c["radio_color"]}!important;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{{color:{c["accent"]}!important;}}
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] + div{{
   color:{c["accent"]}!important;font-weight:600!important;
}}


/* --- SIDEBAR SELECT / TEXT INPUT --- */
[data-testid="stSidebar"] [data-baseweb="select"]>div,
[data-testid="stSidebar"] [data-testid="stTextInput"] input{{
   background:{c["input_bg"]}!important;
   border:1px solid {c["input_border"]}!important;
   color:{c["text_pri"]}!important;border-radius:6px!important;
}}
[data-testid="stSidebar"] [data-baseweb="select"]>div:hover,
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus{{
   border-color:{c["accent"]}!important;
   box-shadow:0 0 0 2px {c["accent_dim"]}!important;
}}
[data-baseweb="popover"],[data-baseweb="popover"] *{{
   background:{c["input_bg"]}!important;color:{c["text_pri"]}!important;
}}


/* --- BUTTONS BASE --- */
.stButton>button{{
   font-weight:700!important;border-radius:7px!important;padding:9px 16px!important;
   font-family:'Exo 2',sans-serif!important;font-size:13px!important;
   letter-spacing:0.5px!important;text-transform:uppercase!important;width:100%!important;
   transition:box-shadow 0.15s ease,transform 0.15s ease!important;
}}


/* --- START button --- */
.st-key-start_btn button{{
   background:{c["btn_start_bg"]}!important;color:{c["btn_start_color"]}!important;
   border:none!important;box-shadow:{c["btn_start_shadow"]}!important;
}}
.st-key-start_btn button:hover{{
   box-shadow:0 6px 20px rgba(14,165,233,0.55)!important;transform:translateY(-1px)!important;
}}
.st-key-start_btn button:disabled{{opacity:0.5!important;cursor:not-allowed!important;}}


/* --- STOP button --- */
.st-key-stop_btn button{{
   background:{c["btn_stop_bg"]}!important;color:{c["btn_stop_color"]}!important;
   border:1px solid {c["btn_stop_border"]}!important;box-shadow:none!important;
}}
.st-key-stop_btn button:disabled{{opacity:0.4!important;cursor:not-allowed!important;}}


/* --- RESET button --- */
.st-key-reset_btn button{{
   background:{c["btn_reset_bg"]}!important;color:{c["btn_reset_color"]}!important;
   border:1px solid {c["btn_reset_border"]}!important;box-shadow:none!important;
}}


/* --- SIDEBAR COLLAPSE BUTTON (arrow toggle) --- */
[data-testid="stSidebarCollapseButton"]{{
   position:fixed!important;top:10px!important;left:10px!important;z-index:999998!important;
}}
[data-testid="stSidebarCollapseButton"] button{{
   background:{c["nav_bg"]}!important;border:1px solid {c["nav_border"]}!important;
   border-radius:6px!important;box-shadow:none!important;
   width:32px!important;height:32px!important;padding:0!important;
   display:flex!important;align-items:center!important;justify-content:center!important;
   overflow:hidden!important;min-width:unset!important;
}}
[data-testid="stSidebarCollapseButton"] button::after{{
   content: '\\25C0';
   font-size: 13px;
   color: {c["nav_text"]};
   font-style: normal;
   display: block;
   line-height: 1;
}}
[data-testid="stSidebarCollapsed"] [data-testid="stSidebarCollapseButton"] button::after{{
   content: '\\25B6';
}}
[data-testid="stSidebarCollapseButton"] button:hover{{
   background:{c["accent_dim"]}!important;border-color:{c["accent"]}!important;
}}
[data-testid="stSidebarCollapseButton"] button:hover::after{{color:{c["accent"]};}}
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stSidebarCollapseButton"] button p,
[data-testid="stSidebarCollapseButton"] svg{{
   display:none!important;width:0!important;height:0!important;overflow:hidden!important;
}}


/* --- THEME TOGGLE (fixed top-right) --- */
.st-key-theme_toggle_btn{{
   position:fixed!important;top:8px!important;right:52px!important;z-index:999999!important;
}}
.st-key-theme_toggle_btn,
.st-key-theme_toggle_btn>div,
.st-key-theme_toggle_btn>div>div{{width:auto!important;display:inline-block!important;}}
.st-key-theme_toggle_btn button{{
   background:{c["nav_bg"]}!important;color:{c["nav_text"]}!important;
   border:1px solid {c["nav_border"]}!important;border-radius:6px!important;
   font-family:'JetBrains Mono',monospace!important;
   font-size:11px!important;font-weight:600!important;letter-spacing:0.5px!important;
   padding:5px 14px!important;box-shadow:none!important;
   text-transform:none!important;white-space:nowrap!important;
   width:auto!important;min-width:unset!important;height:32px!important;line-height:1!important;
}}
.st-key-theme_toggle_btn button:hover{{
   border-color:{c["accent"]}!important;color:{c["accent"]}!important;
   background:{c["accent_dim"]}!important;
}}


/* --- ABOUT BUTTON --- */
.st-key-about_sidebar_btn button{{
   background:{c["accent_dim"]}!important;color:{c["accent"]}!important;
   border:1px solid {c["border"]}!important;border-radius:6px!important;
   font-size:12px!important;font-weight:600!important;letter-spacing:0.3px!important;
   box-shadow:none!important;text-transform:none!important;
   padding:8px 14px!important;width:100%!important;
}}
.st-key-about_sidebar_btn button:hover{{
   border-color:{c["accent"]}!important;background:{c["accent"]}!important;color:#ffffff!important;
}}


/* --- TABS --- */
.stTabs [data-baseweb="tab-list"]{{
   background:transparent!important;border-bottom:1px solid {c["tab_border"]}!important;gap:4px!important;
}}
.stTabs [data-baseweb="tab"]{{
   background:transparent!important;color:{c["tab_color"]}!important;
   font-family:'Exo 2',sans-serif!important;font-size:13px!important;font-weight:600!important;
   border-bottom:2px solid transparent!important;padding:10px 20px!important;
   text-transform:uppercase!important;letter-spacing:0.8px!important;
}}
.stTabs [aria-selected="true"]{{
   color:{c["accent"]}!important;border-bottom:2px solid {c["accent"]}!important;
   background:{c["accent_dim"]}!important;
}}


/* --- DATAFRAME --- */
[data-testid="stDataFrame"]>div{{
   background:{c["df_bg"]}!important;
   border:1px solid {c["df_border"]}!important;border-radius:8px!important;
}}


/* --- SLIDER --- */
div[data-testid="stSlider"] [role="slider"]{{
   box-shadow:none!important;outline:none!important;background:{c["accent"]}!important;
}}


/* --- ALERTS --- */
[data-testid="stAlert"]{{
   background:{c["card_bg"]}!important;
   border:1px solid {c["border"]}!important;border-radius:8px!important;
}}
[data-testid="stAlert"] *{{color:{c["text_sec"]}!important;}}


/* --- DIALOG --- */
[data-testid="stDialog"]>div{{
   background:{c["df_bg"]}!important;
   border:1px solid {c["border"]}!important;border-radius:12px!important;
   box-shadow:0 25px 60px rgba(0,0,0,0.4)!important;
}}


/* --- SCROLLBAR --- */
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{c["page_bg"]};}}
::-webkit-scrollbar-thumb{{background:{c["border"]};border-radius:3px;}}


/* --- RESPONSIVE CHART CONTAINER --- */
.chart-wrapper{{width:100%;overflow-x:auto;}}
</style>"""




st.markdown(_build_css(C), unsafe_allow_html=True)




# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================
_DEFAULT_STATE = {
   "running":            False,
   "csv_index":          0,
   "metrics":            {"evaluated": 0, "safe": 0, "attacks": 0},
   "threat_vectors":     {},
   "telemetry_logs":     [],
   "trend_history":      [],
   "last_attack_vector": None,
   "last_attack_label":  None,
   "xai_background_pool":[],
   "xai_normal_pool":    [],
}
for _k, _v in _DEFAULT_STATE.items():
   if _k not in st.session_state:
       st.session_state[_k] = _v




# =============================================================================
# THEME TOGGLE (fixed top-right corner)
# =============================================================================
_toggle_label = "Light Mode" if st.session_state.theme == "dark" else "Dark Mode"
if st.button(_toggle_label, key="theme_toggle_btn"):
   st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
   st.rerun()




# =============================================================================
# ABOUT DIALOG
# =============================================================================
@st.dialog("About the Research — RT-DeepNIDS")
def _show_abstract():
   c = C
   st.markdown(f"""
<div style="background:{c['card_bg']};border-left:3px solid {c['accent']};
           border-radius:6px;padding:20px 22px;box-shadow:0 4px 20px rgba(0,0,0,0.15);">
 <h4 style="font-family:'Rajdhani',sans-serif;font-size:18px;font-weight:700;
            color:{c['text_pri']};margin-top:0;text-transform:uppercase;letter-spacing:1px;">
   Abstract
 </h4>
 <p style="font-family:'Exo 2',sans-serif;color:{c['text_sec']};font-size:13.5px;line-height:1.75;margin-bottom:0;">
   The rapid expansion of digital communication, IoT architectures, and cloud-based services has led to an unprecedented surge in sophisticated cyberattacks. 
   Traditional signature-based Intrusion Detection Systems (IDS) rely on predefined rules, rendering them ineffective against zero-day vulnerabilities and polymorphic malware while generating excessive false positives. 
   To address these limitations, this paper proposes an intelligent Network Intrusion Detection System (NIDS) leveraging advanced Machine Learning (ML) and Deep Learning (DL) methodologies. 
   The framework evaluates hybrid CNN-GRU and CNN+Transformer architectures alongside Decision Tree, Random Forest, and XGBoost baselines across three benchmark datasets: CIC-IDS-2017, CIC-IDS-2018, and ToN-IoT-v3. 
   Two distinct evaluation pipelines are implemented: an intra-domain pipeline addressing class imbalance via SMOTE and a hybrid Tomek Links with Instance Hardness Threshold (Tomek+IHT) balancing strategy, 
   and a cross-domain transfer learning pipeline where models trained on IT network traffic are deployed directly onto unseen IoT telemetry to assess zero-day generalization. 
   Intra-domain results achieved near-perfect detection accuracies exceeding 99.8%, with Tomek+IHT reducing deep learning training latency by nearly tenfold at less than 2% accuracy trade-off. 
   In the cross-domain evaluation, Random Forest and XGBoost achieved robust accuracies of 96.60% and 96.50%, respectively. 
   SHAP-based Explainable AI (XAI) was integrated to interpret model decisions and identify critical network features driving anomaly detection. 
   The proposed models were further deployed into a lightweight real-time web-based monitoring prototype, demonstrating practical applicability. 
   This research delivers an accurate, explainable, and scalable NIDS solution capable of autonomous threat detection across both IT infrastructures and resource-constrained IoT environments.
 </p>
</div>""", unsafe_allow_html=True)




# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
   c = C
   st.markdown(f"""
<div style="text-align:center;padding-bottom:14px;border-bottom:1px solid {c['border']};margin-bottom:6px;">
 <div style="font-family:'Rajdhani',sans-serif;font-size:19px;font-weight:800;
             color:{c['text_pri']};text-transform:uppercase;letter-spacing:2px;">
   System Control
 </div>
 <div style="font-size:9px;color:{c['accent']};font-weight:700;letter-spacing:2.5px;
             font-family:'JetBrains Mono',monospace;margin-top:3px;">
   MONITORING CONFIGURATION
 </div>
</div>""", unsafe_allow_html=True)


   if st.button("About the Research", key="about_sidebar_btn", use_container_width=True):
       _show_abstract()


   st.markdown("---")


   monitoring_mode = st.radio(
       "Monitoring Mode",
       ["Intra-Domain (Standard)", "Cross-Domain (Robustness Test)"],
       disabled=st.session_state.running,
       key="mode_selector",
   )
   st.markdown("---")


   # Dataset / balancing / model selectors
   if monitoring_mode == "Intra-Domain (Standard)":
       dataset_choice   = st.selectbox(
           "Dataset",
           ["— Select —", "CIC-IDS-2017", "CIC-IDS-2018", "ToN-IoT-v3"],
           disabled=st.session_state.running, key="ds_intra",
       )
       balancing_choice = st.selectbox(
           "Balancing Technique",
           ["— Select —", "SMOTE", "Tomek+IHT"],
           disabled=st.session_state.running, key="bal_intra",
       )
       model_choice = st.selectbox(
           "Model",
           ["— Select —", "Hybrid CNN-GRU", "Random Forest", "XGBoost", "Decision Tree"],
           disabled=st.session_state.running, key="model_intra",
           help="CNN+Transformer is only available in Cross-Domain mode.",
       )
   else:
       st.text_input("Dataset",             value="Source (CIC-IDS) to Target (ToN-IoT)", disabled=True, key="ds_cross")
       st.text_input("Balancing Technique", value="Tomek+IHT",                            disabled=True, key="bal_cross")
       dataset_choice   = "Cross-Domain"
       balancing_choice = "Tomek+IHT"
       model_choice = st.selectbox(
           "Model",
           ["— Select —", "Decision Tree", "Random Forest", "XGBoost", "Hybrid CNN-GRU", "CNN+Transformer"],
           disabled=st.session_state.running, key="model_cross",
       )


   st.markdown("---")


   # Traffic source — CSV Simulation only
   traffic_source = st.selectbox(
       "Traffic Source",
       ["— Select —", "CSV Simulation"],
       disabled=st.session_state.running,
       key="traffic_source_selector",
   )


   st.markdown("---")


   # Configuration readiness guard
   _unselected    = "— Select —"
   _config_ready  = not any(
       v == _unselected for v in [dataset_choice, balancing_choice, model_choice, traffic_source]
   )
   if not _config_ready and not st.session_state.running:
       st.warning("Select all options above before starting.")


   col1, col2 = st.columns(2)
   with col1:
       if st.button(
           "START", use_container_width=True,
           disabled=(st.session_state.running or not _config_ready),
           key="start_btn",
       ):
           st.session_state.running = True
           st.rerun()
   with col2:
       if st.button(
           "STOP", use_container_width=True,
           disabled=not st.session_state.running,
           key="stop_btn",
       ):
           st.session_state.running = False
           st.rerun()


   if st.button("RESET", use_container_width=True, key="reset_btn"):
       # Reset all state keys to defaults
       for _k, _v in _DEFAULT_STATE.items():
           import copy
           st.session_state[_k] = copy.deepcopy(_v)
       st.rerun()


   st.markdown("---")
   throttle_speed   = st.slider("Simulation Speed (sec / packet)", 0.01, 1.0, 0.05, step=0.01)
   render_interval  = st.slider("Refresh Interval (sec)",          0.1,  2.0, 0.5,  step=0.1)
   telemetry_filter = st.radio("Log Filter", ["All Logs", "Normal Only", "Attacks Only"])




# =============================================================================
# BACKEND PATH RESOLUTION & ASSET LOADING
# =============================================================================
current_dataset_dir = resolve_dataset_dir(monitoring_mode, balancing_choice, dataset_choice)
target_model_file   = resolve_model_file(monitoring_mode, model_choice)


target_model_path  = os.path.join(current_dataset_dir, target_model_file) if target_model_file else ""
target_scaler_path = os.path.join(current_dataset_dir, "scaler.pkl")
_model_missing     = bool(target_model_file and not os.path.exists(target_model_path))
_scaler_missing    = not os.path.exists(target_scaler_path)




def _show_asset_guide(missing_model: bool, missing_scaler: bool,
                     model_file: str, dataset_dir: str) -> None:
   """Render an asset-resolution failure card with the expected folder tree."""
   st.markdown(f"""
<div style="background:linear-gradient(145deg,#1a0808,#120606);
           border:1px solid #7f1d1d;border-left:3px solid #ef4444;
           border-radius:10px;padding:20px 24px;margin-bottom:18px;">
 <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
             color:#ef4444;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
   Asset Resolution Failure
 </div>
 <p style="font-family:'Exo 2',sans-serif;font-size:13px;color:#fca5a5;margin:0 0 12px;">
   One or more model asset files could not be located.
   Verify your <code>Real_Time_Export</code> folder matches the structure below.
 </p>
 <div style="font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#f87171;
             background:#0d0303;border:1px solid #7f1d1d;border-radius:6px;
             padding:14px 16px;line-height:1.9;white-space:pre;overflow-x:auto;">
Real_Time_Export/
├── SMOTE/
│   ├── CIC_IDS_2017/
│   │   ├── CNN_GRU_model.h5
│   │   ├── RF_model.pkl  |  DT_model.pkl  |  XGB_model.pkl
│   │   ├── scaler.pkl
│   │   └── live_traffic_sample.csv
│   ├── CIC_IDS_2018/   (same structure)
│   └── TON_IOT_V3/     (same structure)
├── Tomek_IHT/          (same structure as SMOTE/)
└── Cross_Validation/
   ├── CNN_Transformer_model.h5
   ├── CNN_GRU_model.h5
   ├── RF_model.pkl  |  DT_model.pkl  |  XGB_model.pkl
   ├── scaler.pkl
   └── live_traffic_sample.csv
 </div>
 <p style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#fca5a5;margin-top:12px;line-height:1.7;">
   <b>Expected path :</b> {dataset_dir}<br>
   <b>Model file    :</b> {model_file or "(none selected)"}
   {'  —  NOT FOUND' if missing_model  else '  —  Found'}<br>
   <b>Scaler file   :</b> scaler.pkl
   {'  —  NOT FOUND' if missing_scaler else '  —  Found'}
 </p>
</div>""", unsafe_allow_html=True)




@st.cache_resource(show_spinner="Loading model assets...")
def _cached_load(dataset_dir: str, model_filename: str):
   return load_framework_assets(dataset_dir, model_filename)




@st.cache_data(show_spinner=False)
def _cached_csv(csv_path: str):
   return load_csv_source(csv_path)




active_model, active_scaler, classes = _cached_load(current_dataset_dir, target_model_file)
csv_path  = os.path.join(current_dataset_dir, "live_traffic_sample.csv")
df_source = _cached_csv(csv_path)


# Stop immediately if model is missing while running
if active_model is None and st.session_state.running:
   _show_asset_guide(_model_missing, _scaler_missing, target_model_file, current_dataset_dir)
   st.session_state.running = False
   st.stop()


# Passive warning when idle and files are absent
if not st.session_state.running and target_model_file and _model_missing:
   _show_asset_guide(_model_missing, _scaler_missing, target_model_file, current_dataset_dir)


_feature_count = get_feature_count(active_scaler, active_model, dataset_choice)


PLOT_BG = "rgba(0,0,0,0)"




def _base_layout(c: dict, **kw) -> dict:
   return dict(
       paper_bgcolor=PLOT_BG,
       plot_bgcolor=PLOT_BG,
       font=dict(family="'JetBrains Mono', monospace", color=c["tick"], size=11),
       margin=dict(l=10, r=40, t=35, b=10),
       uirevision="constant",
       **kw,
   )




# =============================================================================
# LANDING PAGE (shown before first run)
# =============================================================================
def _landing_page_html(c: dict) -> str:
   return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@700&family=JetBrains+Mono:wght@600;700&family=Exo+2:wght@400;600;700;800&display=swap');
.lp-wrap{{
   background:{c['card_bg']};border:1px solid {c['card_border']};
   border-top:2px solid {c['card_border_t']};border-radius:14px;
   padding:40px 44px 36px;margin-top:12px;
   box-shadow:{c['card_shadow']};position:relative;overflow:hidden;
}}
.lp-wrap::before{{
   content:'';position:absolute;top:0;left:0;right:0;height:1px;
   background:linear-gradient(90deg,transparent,{c['accent']},transparent);opacity:0.5;
}}
.lp-title{{
   font-family:'Rajdhani',sans-serif;font-size:28px;font-weight:700;
   color:{c['text_pri']};text-align:center;margin-bottom:4px;
   text-transform:uppercase;letter-spacing:2px;
}}
.lp-subtitle{{
   font-family:'Exo 2',sans-serif;font-size:13px;color:{c['accent']};
   text-align:center;letter-spacing:1px;margin-bottom:6px;font-weight:600;
}}
.lp-inst{{
   font-family:'Exo 2',sans-serif;font-size:13px;color:{c['text_muted']};
   text-align:center;margin-bottom:32px;line-height:1.6;
}}
.lp-inst b{{color:{c['accent']};}}
.lp-sec{{
   font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
   color:{c['accent']};text-transform:uppercase;letter-spacing:2px;
   border-bottom:1px solid {c['border']};padding-bottom:8px;margin:26px 0 14px;
}}
.team-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;}}
.team-card{{
   background:{c['card_bg']};border:1px solid {c['card_border']};
   border-radius:9px;padding:14px;text-align:center;box-shadow:{c['card_shadow']};
}}
.team-name{{font-family:'Exo 2',sans-serif;font-weight:700;color:{c['text_pri']};font-size:13.5px;margin-bottom:4px;}}
.team-id{{font-family:'JetBrains Mono',monospace;font-size:11px;color:{c['accent']};font-weight:600;}}
.pipeline{{
   display:flex;justify-content:space-between;align-items:center;
   background:{c['threat_box_bg']};border:1px solid {c['border']};border-radius:10px;
   padding:20px 22px;flex-wrap:wrap;gap:8px;
}}
.p-step{{text-align:center;flex:1;min-width:100px;}}
.p-num{{
   width:26px;height:26px;background:linear-gradient(145deg,#0ea5e9,#0284c7);
   border-radius:50%;display:flex;align-items:center;justify-content:center;
   font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;
   color:white;margin:0 auto 7px;box-shadow:0 0 10px rgba(14,165,233,0.3);
}}
.p-title{{font-family:'Exo 2',sans-serif;font-size:12px;font-weight:700;color:{c['text_pri']};margin-bottom:2px;}}
.p-desc{{font-family:'Exo 2',sans-serif;font-size:11px;color:{c['text_muted']};}}
.p-arrow{{color:{c['border']};font-size:18px;font-weight:bold;}}
.lp-table{{
   width:100%;border-collapse:collapse;margin-top:8px;
   border:1px solid {c['border']};border-radius:9px;overflow:hidden;
   box-shadow:{c['card_shadow']};
}}
.lp-table th{{
   background:{c['card_bg']};color:{c['accent']};padding:12px 16px;text-align:left;
   font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
   letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid {c['border']};
}}
.lp-table td{{
   padding:11px 16px;border-bottom:1px solid {c['border']};
   font-family:'Exo 2',sans-serif;font-size:13px;color:{c['text_sec']};
}}
.lp-table tr:last-child td{{border-bottom:none;}}
.badge{{
   display:inline-block;background:{c['accent_dim']};border:1px solid {c['border']};
   color:{c['accent']};padding:2px 8px;border-radius:4px;
   font-size:11px;font-family:'JetBrains Mono',monospace;font-weight:600;
}}
</style>
<div class="lp-wrap">
 <div class="lp-title">RT-DeepNIDS</div>
 <div class="lp-subtitle">A Real-Time Hybrid Network Intrusion Detection System</div>
 <div class="lp-inst">
   Bangladesh University of Business and Technology (BUBT)<br>
   <b>Department of Computer Science and Engineering</b>
 </div>


 <div class="lp-sec">Research Team</div>
 <div class="team-grid">
   <div class="team-card"><div class="team-name">Ayesha Siddika</div><div class="team-id">22234103099</div></div>
   <div class="team-card"><div class="team-name">Sanjida Khanom</div><div class="team-id">22234103103</div></div>
   <div class="team-card"><div class="team-name">Ihsanul Hossain Rafsan</div><div class="team-id">22234103112</div></div>
   <div class="team-card"><div class="team-name">Sadia Mehrin Rahi</div><div class="team-id">22234103122</div></div>
   <div class="team-card"><div class="team-name">Istiyak Hasan Maruf</div><div class="team-id">22234103130</div></div>
 </div>


 <div class="lp-sec">System Methodology Pipeline</div>
 <div class="pipeline">
   <div class="p-step">
     <div class="p-num">01</div>
     <div class="p-title">Data Ingestion</div>
     <div class="p-desc">Raw PCAP / CSV</div>
   </div>
   <div class="p-arrow">&#8250;</div>
   <div class="p-step">
     <div class="p-num">02</div>
     <div class="p-title">Preprocessing</div>
     <div class="p-desc">Cleaning and Scaling</div>
   </div>
   <div class="p-arrow">&#8250;</div>
   <div class="p-step">
     <div class="p-num">03</div>
     <div class="p-title">Balancing</div>
     <div class="p-desc">SMOTE / Tomek+IHT</div>
   </div>
   <div class="p-arrow">&#8250;</div>
   <div class="p-step">
     <div class="p-num">04</div>
     <div class="p-title">Model Training</div>
     <div class="p-desc">ML and DL Architectures</div>
   </div>
   <div class="p-arrow">&#8250;</div>
   <div class="p-step">
     <div class="p-num">05</div>
     <div class="p-title">Prediction</div>
     <div class="p-desc">Real-Time Telemetry</div>
   </div>
 </div>


 <div class="lp-sec">Dataset Benchmarks</div>
 <table class="lp-table">
   <thead>
     <tr><th>Dataset</th><th>Feature Space</th><th>Total Samples (Approx.)</th></tr>
   </thead>
   <tbody>
     <tr>
       <td><strong>CIC-IDS-2017</strong></td>
       <td><span class="badge">78 Features</span></td>
       <td>2.8 M Samples</td>
     </tr>
     <tr>
       <td><strong>CIC-IDS-2018</strong></td>
       <td><span class="badge">80 Features</span></td>
       <td>16 M Samples</td>
     </tr>
     <tr>
       <td><strong>ToN-IoT-v3</strong></td>
       <td><span class="badge">~470 Features</span></td>
       <td>2.4 M Samples</td>
     </tr>
   </tbody>
 </table>
</div><br>"""




# Show landing page only on first load
if (not st.session_state.running
       and st.session_state.csv_index == 0
       and not st.session_state.telemetry_logs):
   st.info("Select your configuration in the control panel and press Start to begin monitoring.")
   st.markdown(_landing_page_html(C), unsafe_allow_html=True)
   st.stop()




# =============================================================================
# XAI / SHAP PANEL
# =============================================================================
def _render_xai_panel(c: dict, model, scaler, m_choice: str,
                     ds_choice: str, n_feats: int) -> None:
   """
   Render the Explainable AI panel for the most recent Attack prediction.
   Uses SHAP when available; falls back to sklearn feature_importances_.
   """
   st.markdown(f"""
<div style="background:{c['threat_box_bg']};border:1px solid {c['border']};
           border-left:3px solid {c['accent']};border-radius:10px;
           padding:16px 20px;margin-bottom:14px;">
 <div style="font-size:10px;font-weight:700;color:{c['accent']};
             font-family:'JetBrains Mono',monospace;letter-spacing:2px;margin-bottom:6px;">
   EXPLAINABLE AI — DECISION TRANSPARENCY (SHAP)
 </div>
 <div style="font-size:12px;color:{c['text_sec']};font-family:'Exo 2',sans-serif;line-height:1.6;">
   SHapley Additive exPlanations — top features that drove the most recent
   <b style="color:#ef4444;">Attack</b> classification decision.
 </div>
</div>""", unsafe_allow_html=True)


   if model is None:
       st.warning("Model not loaded. Select a valid configuration and press Start, then enable XAI.")
       return


   # Resolve feature column labels from DATASET_COLS
   _raw_cols   = DATASET_COLS.get(ds_choice, [])
   _col_labels = [str(col).strip()[:35] for col in _raw_cols]
   while len(_col_labels) < n_feats:
       _col_labels.append(f"feat_{len(_col_labels)}")


   explain_vec = st.session_state.get("last_attack_vector", None)
   last_label  = st.session_state.get("last_attack_label",  "Attack")


   if explain_vec is None:
       st.info(
           "No attack prediction recorded yet. Start monitoring — "
           "the XAI panel will populate on the first attack detection."
       )
       return


   explain_vec = np.array(explain_vec, dtype=float).flatten()
   explain_vec = (
       np.pad(explain_vec, (0, n_feats - len(explain_vec)))
       if len(explain_vec) < n_feats
       else explain_vec[:n_feats]
   )
   explain_2d = explain_vec.reshape(1, -1)


   # Build SHAP background from Normal-only pool (or fallback to mixed pool)
   pool      = st.session_state.get("xai_background_pool", [])
   norm_pool = st.session_state.get("xai_normal_pool",     [])


   if len(norm_pool) >= 5:
       _bg_rows  = norm_pool[-100:]
       _bg_arr   = np.array(
           [np.pad(r, (0, max(0, n_feats - len(r))))[:n_feats] for r in _bg_rows],
           dtype=float,
       )
       try:
           import shap as _shap_bg
           _k         = min(10, len(_bg_arr))
           background = _shap_bg.kmeans(_bg_arr, _k)
           _bg_label  = f"shap.kmeans(k={_k}, {len(_bg_arr)} Normal vectors)"
       except Exception:
           background = _bg_arr
           _bg_label  = f"Raw Normal background ({len(_bg_arr)} rows)"
   elif len(pool) >= 5:
       _bg_rows  = pool[-30:]
       background = np.array(
           [np.pad(r, (0, max(0, n_feats - len(r))))[:n_feats] for r in _bg_rows],
           dtype=float,
       )
       _bg_label = f"Mixed pool fallback ({len(_bg_rows)} rows)"
   else:
       background = explain_2d
       _bg_label  = "Explain-vec fallback (pool too small)"


   _DL           = {"Hybrid CNN-GRU", "CNN+Transformer"}
   _shap_vals    = None
   _imp_vals     = None
   _method_label = "N/A"
   _has_shap     = False


   try:
       import shap as _shap
       _has_shap = True
   except ImportError:
       pass


   if _has_shap:
       try:
           if m_choice in _DL:
               def _pred_fn(x):
                   inp = x.reshape(x.shape[0], 1, x.shape[1])
                   out = model.predict(inp, verbose=0)
                   return out if out.shape[-1] > 1 else np.hstack([1 - out, out])


               _exp          = _shap.KernelExplainer(_pred_fn, background)
               _raw          = _exp.shap_values(explain_2d, nsamples=128, silent=True)
               _method_label = f"KernelSHAP (DL)  |  background: {_bg_label}"
           else:
               _exp          = _shap.TreeExplainer(model)
               _raw          = _exp.shap_values(explain_2d)
               _method_label = "TreeSHAP (exact)"


           # Normalise SHAP output shape
           if isinstance(_raw, list):
               _raw = _raw[-1]
           _raw       = np.array(_raw, dtype=float).squeeze()
           if _raw.ndim == 2: _raw = _raw[0]
           if _raw.ndim != 1: _raw = _raw.flatten()
           _shap_vals = np.abs(_raw).astype(float)


       except Exception as _shap_e:
           st.warning(f"SHAP computation failed: {_shap_e}  —  using feature_importances_ fallback.")
           if hasattr(model, "feature_importances_"):
               _imp_vals     = np.array(model.feature_importances_, dtype=float).flatten()
               _method_label = f"Feature Importance (SHAP error: {str(_shap_e)[:60]})"
           else:
               st.error(f"SHAP failed and model has no feature_importances_.\n{_shap_e}")
               return
   else:
       if hasattr(model, "feature_importances_"):
           _imp_vals     = np.array(model.feature_importances_, dtype=float).flatten()
           _method_label = "Feature Importance (sklearn)  —  install shap for SHAP values"
           st.info("shap not installed. Showing sklearn feature_importances_. Run: pip install shap")
       else:
           st.error("shap not installed and this model has no feature_importances_. Run: pip install shap")
           return


   _importance = _shap_vals if _shap_vals is not None else _imp_vals
   if _importance is None:
       st.error("No importance values available.")
       return


   _importance = np.array(_importance, dtype=float).flatten()
   n_avail     = min(len(_importance), len(_col_labels))
   _importance = _importance[:n_avail]
   _names      = np.array(_col_labels[:n_avail], dtype=object)


   top_n   = min(15, n_avail)
   top_idx = np.argsort(_importance)[::-1][:top_n]
   t_vals  = _importance[top_idx].astype(float)
   t_names = [str(s) for s in _names[top_idx].tolist()]
   _max    = float(t_vals.max()) if t_vals.max() > 0 else 1.0
   t_norm  = (t_vals / _max).tolist()


   clrs = [
       f"rgba(239,68,68,{0.4 + 0.6*v:.2f})" if v > 0.5
       else f"rgba(56,189,248,{0.3 + 0.5*v:.2f})"
       for v in t_norm
   ]
   explain_vals = [f"{float(explain_vec[i]):.4f}" for i in top_idx.tolist()]


   fig_xai = go.Figure(go.Bar(
       x=t_vals[::-1].tolist(),
       y=[n[:32] for n in t_names[::-1]],
       orientation="h",
       marker=dict(color=clrs[::-1], line=dict(color="rgba(0,0,0,0)", width=0)),
       text=[f"{float(v):.4f}" for v in t_vals[::-1]],
       textposition="outside",
       textfont=dict(family="JetBrains Mono", size=10, color=c["text_sec"]),
       customdata=explain_vals[::-1],
       hovertemplate=(
           "<b>%{y}</b><br>"
           "SHAP |value|: %{x:.4f}<br>"
           "Feature value: %{customdata}<extra></extra>"
       ),
   ))
   fig_xai.update_layout(
       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
       font=dict(family="'JetBrains Mono',monospace", color=c["tick"], size=11),
       margin=dict(l=10, r=60, t=40, b=10),
       uirevision="xai_chart",
       height=max(360, top_n * 28),
       title=dict(
           text=f"Top {top_n} Features  |  Label: <b>{last_label}</b>  |  {_method_label}",
           font=dict(size=11, color=c["text_sec"], family="Exo 2"),
           x=0, xanchor="left",
       ),
       xaxis=dict(
           showgrid=True, gridcolor=c["grid"], zeroline=False,
           title=dict(text="SHAP |value| / Importance",
                      font=dict(size=11, color=c["text_muted"], family="JetBrains Mono")),
           tickfont=dict(color=c["tick"], family="JetBrains Mono"),
       ),
       yaxis=dict(
           showgrid=False, type="category", automargin=True,
           tickfont=dict(color=c["tick"], family="JetBrains Mono", size=11),
       ),
       bargap=0.3,
   )
   st.plotly_chart(fig_xai, use_container_width=True, theme=None,
                   config={"displayModeBar": False}, key="xai_chart")


   if t_names:
       top1_name = t_names[0]
       top1_val  = float(explain_vec[int(top_idx[0])]) if int(top_idx[0]) < len(explain_vec) else 0.0
       st.markdown(f"""
<div style="font-size:12px;color:{c['text_muted']};font-family:'Exo 2',sans-serif;
           padding:10px 14px;background:{c['accent_dim']};border-radius:6px;margin-top:6px;">
 <b style="color:{c['accent']};">Interpretation:</b>
 Feature <code style="color:{c['text_pri']};font-size:11px;">{top1_name}</code>
 (value <code style="color:{c['text_pri']};">{top1_val:.4f}</code>)
 contributed most strongly to classifying this flow as
 <b style="color:#ef4444;">{last_label}</b>.
 &nbsp;|&nbsp;
 <span style="color:#ef4444;">Red bars</span> = high attack influence &nbsp;
 <span style="color:#38bdf8;">Blue bars</span> = lower influence.
 &nbsp;|&nbsp; Method: <b>{_method_label}</b>
</div>""", unsafe_allow_html=True)




# =============================================================================
# LIVE DASHBOARD FRAGMENT
# =============================================================================
@st.fragment(run_every=render_interval if st.session_state.running else None)
def _render_live_dashboard() -> None:
   c             = T_DARK if st.session_state.theme == "dark" else T_LIGHT
   feature_count = _feature_count


   # -------------------------------------------------------------------------
   # BATCH PROCESSING
   # -------------------------------------------------------------------------
   if st.session_state.running:


       # --- CSV Simulation ---------------------------------------------------
       if df_source is not None:
           batch_size   = max(1, int(render_interval / max(throttle_speed, 0.001)))
           feature_cols = [col for col in df_source.columns if col not in EXCLUDE_COLS]
           ts           = datetime.now().strftime("%H:%M:%S")
           batch_n = batch_a = 0


           for _ in range(batch_size):
               if st.session_state.csv_index >= len(df_source):
                   st.session_state.running = False
                   break


               row = df_source.iloc[st.session_state.csv_index]
               raw = row[feature_cols].values.astype(float)
               n   = min(len(raw), feature_count)
               feat = raw[:n].reshape(1, -1)
               if n < feature_count:
                   feat = np.pad(feat, ((0, 0), (0, feature_count - n)))


               scaled = feat
               if active_scaler:
                   try:    scaled = active_scaler.transform(feat)
                   except: scaled = feat


               pred_class = 0
               if model_choice in DL_MODELS:
                   try:
                       inp  = scaled.reshape((1, 1, scaled.shape[1]))
                       prob = active_model.predict(inp, verbose=0)
                       pred_class = (
                           int(prob[0][0] > 0.5) if prob.shape[-1] == 1
                           else int(np.argmax(prob, axis=1)[0])
                       )
                   except: pred_class = 0
               else:
                   try:    pred_class = int(active_model.predict(scaled)[0])
                   except: pred_class = 0


               label = (
                   clean_label(str(classes[pred_class]))
                   if classes is not None and 0 <= pred_class < len(classes)
                   else ("Normal" if pred_class == 0 else "Attack")
               )


               src_ip = row.get("Source IP",
                   f"10.0.{np.random.randint(0,255)}.{np.random.randint(1,254)}")
               port  = safe_int(row.get("Dest Port", row.get(" Destination Port", 80)), 80)
               proto = safe_int(row.get("Protocol", 6), 6)


               _sv = scaled.flatten()
               pool = st.session_state.xai_background_pool
               pool.append(_sv)
               if len(pool) > 100: pool.pop(0)


               st.session_state.metrics["evaluated"] += 1
               if label == "Normal":
                   st.session_state.metrics["safe"] += 1
                   batch_n += 1
                   np_pool = st.session_state.xai_normal_pool
                   np_pool.append(_sv.copy())
                   if len(np_pool) > 100: np_pool.pop(0)
               else:
                   st.session_state.metrics["attacks"] += 1
                   batch_a += 1
                   tv = st.session_state.threat_vectors
                   tv[label] = tv.get(label, 0) + 1
                   st.session_state.last_attack_vector = _sv.copy()
                   st.session_state.last_attack_label  = label


               st.session_state.telemetry_logs.append({
                   "Time": ts, "Source IP": src_ip,
                   "Port": port, "Protocol": proto, "Status": label,
               })
               st.session_state.csv_index += 1


           st.session_state.trend_history.append(
               {"Time": ts, "Normal": batch_n, "Attack": batch_a}
           )



   # -------------------------------------------------------------------------
   # DASHBOARD RENDERING
   # -------------------------------------------------------------------------
   m_total  = st.session_state.metrics["evaluated"]
   m_safe   = st.session_state.metrics["safe"]
   m_attack = st.session_state.metrics["attacks"]
   m_ratio  = f"{(m_attack / m_total * 100):.1f}%" if m_total > 0 else "0.0%"
   progress = (
       st.session_state.csv_index
       / max(len(df_source) if df_source is not None else 1, 1)
       * 100
   )


   current_dt   = datetime.now().strftime("%d %b %Y,  %I:%M:%S %p")
   is_running   = st.session_state.running
   status_color = c["accent"] if is_running else c["text_muted"]
   status_label = "MONITORING ACTIVE" if is_running else "SYSTEM PAUSED"
   source_badge = "CSV Simulation"


   # --- Header ---
   st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start;
           flex-wrap:wrap;gap:12px;margin-bottom:14px;">
 <div style="min-width:0;flex:1;">
   <h1 style="font-family:'Rajdhani',sans-serif;font-size:24px;font-weight:800;
              color:{c['text_pri']};margin:0;text-transform:uppercase;letter-spacing:2px;">
     Real-Time Traffic Monitoring
   </h1>
   <p style="margin:5px 0 0;font-size:12px;color:{c['text_muted']};
             font-family:'Exo 2',sans-serif;line-height:1.6;flex-wrap:wrap;">
     <b style="color:{c['text_sec']};">Mode:</b> {monitoring_mode} &nbsp;&middot;&nbsp;
     <b style="color:{c['text_sec']};">Dataset:</b> {dataset_choice} &nbsp;&middot;&nbsp;
     <b style="color:{c['text_sec']};">Model:</b> {model_choice} &nbsp;&middot;&nbsp;
     <b style="color:{c['text_sec']};">Balancing:</b> {balancing_choice} &nbsp;&middot;&nbsp;
     <b style="color:{c['text_sec']};">Source:</b> {source_badge}
   </p>
 </div>
 <div style="text-align:right;flex-shrink:0;padding-top:2px;">
   <div style="font-size:11px;color:{c['text_muted']};
               font-family:'JetBrains Mono',monospace;margin-bottom:6px;">
     {current_dt}
   </div>
   <span style="background:{c['accent_dim']};border:1px solid {status_color}55;
                color:{status_color};padding:5px 14px;border-radius:20px;
                font-family:'JetBrains Mono',monospace;font-weight:700;
                font-size:10px;letter-spacing:1px;">
     {status_label}
   </span>
 </div>
</div>""", unsafe_allow_html=True)


   # --- KPI Cards ---
   kpi1, kpi2, kpi3, kpi4 = st.columns(4)
   kpi1.metric("Total Packets Evaluated", f"{m_total:,}")
   kpi2.metric("Normal Traffic",          f"{m_safe:,}")
   kpi3.metric("Attack Traffic",          f"{m_attack:,}")
   kpi4.metric("Attack Ratio",            m_ratio)


   # --- Progress bar ---
   st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:14px 0;">
 <span style="font-size:9px;font-weight:700;color:{c['text_muted']};
              font-family:'JetBrains Mono',monospace;letter-spacing:1.5px;
              white-space:nowrap;width:160px;">
   PROCESSING PROGRESS
 </span>
 <div style="flex:1;height:5px;background:{c['progress_track']};
             border-radius:3px;overflow:hidden;min-width:60px;">
   <div style="width:{progress:.1f}%;height:100%;
               background:linear-gradient(90deg,#0ea5e9,#38bdf8,#7dd3fc);
               box-shadow:0 0 8px rgba(56,189,248,0.5);"></div>
 </div>
 <span style="font-size:12px;font-weight:800;color:{c['accent']};
              width:46px;text-align:right;font-family:'JetBrains Mono',monospace;">
   {progress:.1f}%
 </span>
</div>""", unsafe_allow_html=True)


   # --- Threat signature badges ---
   badge_parts = []
   for k, v in st.session_state.threat_vectors.items():
       safe_k = html_mod.escape(str(k))
       badge_parts.append(
           f'<div style="background:rgba(220,38,38,0.08);color:#ef4444;'
           f'border:1px solid rgba(220,38,38,0.25);padding:5px 10px;border-radius:6px;'
           f'font-size:12px;font-weight:600;display:inline-flex;align-items:center;'
           f'margin:0 6px 8px 0;font-family:Exo 2,sans-serif;">'
           f'{safe_k}'
           f'<span style="background:#dc2626;color:white;border-radius:4px;'
           f'padding:2px 7px;margin-left:8px;font-size:11px;'
           f'font-family:JetBrains Mono,monospace;">{v}</span></div>'
       )
   badges_html = "".join(badge_parts)
   empty_msg = (
       f'<span style="font-size:13px;color:{c["text_muted"]};font-style:italic;">'
       f'Scanning data stream for threat signatures...</span>'
   )
   st.markdown(f"""
<div style="background:{c['threat_box_bg']};border:1px solid {c['border']};
           border-radius:10px;padding:14px 18px;margin-bottom:16px;
           box-shadow:inset 0 2px 8px rgba(0,0,0,0.06);">
 <div style="font-size:9px;font-weight:700;color:{c['accent']};
             font-family:'JetBrains Mono',monospace;letter-spacing:2px;margin-bottom:10px;">
   DETECTED THREAT SIGNATURES
 </div>
 <div style="min-height:42px;display:flex;flex-wrap:wrap;align-content:flex-start;">
   {badges_html if badges_html else empty_msg}
 </div>
</div>""", unsafe_allow_html=True)


   # --- Charts ---
   g1, g2 = st.columns([1, 1.8])


   with g1:
       st.markdown(
           f'<div style="font-size:12px;font-weight:700;color:{c["text_sec"]};'
           f'font-family:JetBrains Mono,monospace;letter-spacing:1px;'
           f'text-transform:uppercase;margin-bottom:4px;">Threat Classification</div>',
           unsafe_allow_html=True,
       )
       if m_total > 0:
           labels = ["Normal"] + list(st.session_state.threat_vectors.keys())
           vals   = [m_safe]   + list(st.session_state.threat_vectors.values())
           colors = ["#059669"] + ["#dc2626"] * len(st.session_state.threat_vectors)
           fig_bar = go.Figure(go.Bar(
               x=vals, y=labels, orientation="h",
               marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.1)", width=1), opacity=0.9),
               text=vals, textposition="outside",
               textfont=dict(family="JetBrains Mono", size=11, color=c["text_sec"]),
           ))
           fig_bar.update_layout(
               **_base_layout(c, height=290),
               xaxis=dict(showgrid=True, gridcolor=c["grid"], showticklabels=False, zeroline=False),
               yaxis=dict(showgrid=False, type="category", automargin=True,
                          tickfont=dict(color=c["tick"], family="JetBrains Mono", size=11)),
               bargap=0.35,
           )
           st.plotly_chart(fig_bar, use_container_width=True, theme=None,
                           config={"displayModeBar": False}, key="bar_chart")
       else:
           st.info("Awaiting traffic data...")


   with g2:
       st.markdown(
           f'<div style="font-size:12px;font-weight:700;color:{c["text_sec"]};'
           f'font-family:JetBrains Mono,monospace;letter-spacing:1px;'
           f'text-transform:uppercase;margin-bottom:4px;">Live Traffic Activity</div>',
           unsafe_allow_html=True,
       )
       if st.session_state.trend_history:
           df_trend = pd.DataFrame(st.session_state.trend_history).tail(45)
           fig_line = go.Figure()
           fig_line.add_trace(go.Scatter(
               x=df_trend["Time"], y=df_trend["Normal"], name="Normal",
               mode="lines", line=dict(color="#059669", width=2),
               fill="tozeroy", fillcolor="rgba(5,150,105,0.07)",
           ))
           fig_line.add_trace(go.Scatter(
               x=df_trend["Time"], y=df_trend["Attack"], name="Attack",
               mode="lines", line=dict(color="#dc2626", width=2),
               fill="tozeroy", fillcolor="rgba(220,38,38,0.07)",
           ))
           fig_line.update_layout(
               **_base_layout(c, height=290),
               yaxis=dict(showgrid=True, gridcolor=c["grid"], zeroline=False,
                          tickfont=dict(color=c["tick"], family="JetBrains Mono")),
               xaxis=dict(showgrid=False, tickangle=-30, nticks=10,
                          tickfont=dict(color=c["tick"], family="JetBrains Mono")),
               legend=dict(
                   orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1,
                   font=dict(color=c["text_sec"], family="JetBrains Mono", size=11),
                   bgcolor="rgba(0,0,0,0)",
               ),
           )
           st.plotly_chart(fig_line, use_container_width=True, theme=None,
                           config={"displayModeBar": False}, key="line_chart")
       else:
           st.info("Awaiting time-series data...")


   # --- Log tables ---
   st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
   tab1, tab2 = st.tabs(["Simulated Traffic Logs", "Security Incident Alerts"])


   if st.session_state.telemetry_logs:
       df_logs = pd.DataFrame(st.session_state.telemetry_logs)
       with tab1:
           display = df_logs.copy()
           if telemetry_filter == "Normal Only":
               display = display[display["Status"].str.upper() == "NORMAL"]
           elif telemetry_filter == "Attacks Only":
               display = display[display["Status"].str.upper() != "NORMAL"]
           st.dataframe(
               display.tail(15).style.map(
                   lambda v: (
                       "color:#ef4444;font-weight:700;"
                       if isinstance(v, str) and v.upper() != "NORMAL"
                       else "color:#059669;font-weight:600;"
                   ),
                   subset=["Status"],
               ),
               height=380, use_container_width=True, hide_index=True, key="stream_table",
           )
       with tab2:
           df_alerts = df_logs[df_logs["Status"].str.upper() != "NORMAL"].copy()
           if not df_alerts.empty:
               df_alerts = df_alerts.rename(columns={"Status": "Signature", "Source IP": "IP"})
               df_alerts["Severity"] = "CRITICAL"
               st.dataframe(
                   df_alerts[["Time","Signature","Severity","IP","Port","Protocol"]].tail(15)
                   .style.map(
                       lambda v: (
                           "background:rgba(220,38,38,0.12);color:#ef4444;font-weight:700;"
                           if v == "CRITICAL" else ""
                       ),
                       subset=["Severity"],
                   ),
                   height=380, use_container_width=True, hide_index=True, key="alert_table",
               )
           else:
               st.success("No security incidents detected in the current session.")
   else:
       with tab1:
           st.info("No log entries recorded. Start monitoring to capture traffic.")
       with tab2:
           st.info("No alerts detected. System is ready.")


   # --- XAI Panel ---
   st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
   xai_enabled = st.checkbox(
       "Enable Decision Transparency (XAI / SHAP)",
       value=False,
       key="xai_toggle",
       help="Shows the top features that drove the most recent Attack classification "
            "using SHAP values or sklearn feature_importances_.",
   )
   if xai_enabled:
       _render_xai_panel(
           c, active_model, active_scaler,
           model_choice, dataset_choice, feature_count,
       )




# =============================================================================
# ENTRY POINT
# =============================================================================
_render_live_dashboard()