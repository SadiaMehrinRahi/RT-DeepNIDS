# =============================================================================
# RT-DeepNIDS — backend.py
# A Real-Time Hybrid Network Intrusion Detection System
# Bangladesh University of Business and Technology (BUBT)
# Department of Computer Science and Engineering
#
# Research Team:
#   Ayesha Siddika         (22234103099)
#   Sanjida Khanom         (22234103103)
#   Ihsanul Hossain Rafsan (22234103112)
#   Sadia Mehrin Rahi      (22234103122)
#   Istiyak Hasan Maruf    (22234103130)
#
# This module contains all backend logic:
#   - Dataset column templates (CIC-IDS-2017, CIC-IDS-2018, ToN-IoT-v3)
#   - Model / scaler / CSV loaders
#   - Label cleaning utilities
# =============================================================================


import os
import re


import numpy as np
import pandas as pd
import joblib




# ---------------------------------------------------------------------------
# Directory & model mappings
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "Real_Time_Export")


DATASET_FOLDER_MAP = {
   "CIC-IDS-2017": "CIC_IDS_2017",
   "CIC-IDS-2018": "CIC_IDS_2018",
   "ToN-IoT-v3":   "TON_IOT_V3",
}
BALANCING_FOLDER_MAP = {
   "SMOTE":     "SMOTE",
   "Tomek+IHT": "Tomek_IHT",
   "Tomek_IHT": "Tomek_IHT",   # accept both spellings defensively
}
MODEL_FILE_MAP = {
   "Hybrid CNN-GRU":  "CNN_GRU_model.h5",
   "Decision Tree":   "DT_model.pkl",
   "Random Forest":   "RF_model.pkl",
   "XGBoost":         "XGB_model.pkl",
   "CNN+Transformer": "CNN_Transformer_model.h5",
}
DL_MODELS = {"Hybrid CNN-GRU", "CNN+Transformer"}


EXCLUDE_COLS = {
   "Label", "Class", "Attack", "type", "category",
   "Timestamp", "Source IP", "Dest IP",
   "source_ip", "dest_ip", "src_ip", "dst_ip",
}




# ---------------------------------------------------------------------------
# Dataset column templates
# ---------------------------------------------------------------------------
# DESIGN NOTE:
#   Feature POSITION matters — CICFlowMeter exports have a strict column order.
#   Each template maps column_name -> vector_index; values are placed at those
#   exact indices so the trained scaler / model sees the correct layout.


_CIC2017_COLS = [
   " Destination Port", " Flow Duration", " Total Fwd Packets",
   " Total Backward Packets", "Total Length of Fwd Packets",
   " Total Length of Bwd Packets", " Fwd Packet Length Max",
   " Fwd Packet Length Min", " Fwd Packet Length Mean",
   " Fwd Packet Length Std", "Bwd Packet Length Max",
   " Bwd Packet Length Min", " Bwd Packet Length Mean",
   " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s",
   " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min",
   "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max",
   " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std",
   " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags",
   " Fwd URG Flags", " Bwd URG Flags", " Fwd Header Length",
   " Bwd Header Length", "Fwd Packets/s", " Bwd Packets/s",
   " Min Packet Length", " Max Packet Length", " Packet Length Mean",
   " Packet Length Std", " Packet Length Variance", "FIN Flag Count",
   " SYN Flag Count", " RST Flag Count", " PSH Flag Count",
   " ACK Flag Count", " URG Flag Count", " CWE Flag Count",
   " ECE Flag Count", " Down/Up Ratio", " Average Packet Size",
   " Avg Fwd Segment Size", " Avg Bwd Segment Size",
   " Fwd Header Length.1", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk",
   " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk",
   "Bwd Avg Bulk Rate", "Subflow Fwd Packets", " Subflow Fwd Bytes",
   " Subflow Bwd Packets", " Subflow Bwd Bytes", "Init_Win_bytes_forward",
   " Init_Win_bytes_backward", " act_data_pkt_fwd",
   " min_seg_size_forward", "Active Mean", " Active Std",
   " Active Max", " Active Min", "Idle Mean", " Idle Std",
   " Idle Max", " Idle Min",
]  # 78 features


_CIC2018_COLS = [
   "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
   "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
   "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
   "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
   "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
   "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
   "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
   "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
   "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
   "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
   "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
   "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Cnt", "ECE Flag Cnt",
   "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
   "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
   "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
   "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
   "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
   "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
   "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]  # 80 features


# ToN-IoT-v3 core Zeek/Bro flow fields (47).
# The full ~470-column space is built by _expand_ton_cols() below,
# which appends one-hot / label-encoded expansion columns to match
# the training pipeline's pd.get_dummies() output.
_TON_COLS_CORE = [
   "duration", "proto", "src_port", "dst_port", "src_pkts", "src_bytes",
   "dst_pkts", "dst_bytes", "conn_state", "missed_bytes", "src_ip_bytes",
   "dst_ip_bytes", "dns_query", "dns_qclass", "dns_qtype", "dns_rcode",
   "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
   "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
   "ssl_subject", "ssl_issuer",
   "http_trans_depth", "http_method", "http_uri", "http_referrer",
   "http_version", "http_request_body_len", "http_response_body_len",
   "http_status_code", "http_user_agent", "http_orig_mime_types",
   "http_resp_mime_types",
   "weird_name", "weird_addl", "weird_notice",
   "service", "duration_log1p", "src_bytes_log1p", "dst_bytes_log1p",
   "src_pkts_log1p", "dst_pkts_log1p", "missed_bytes_log1p",
]  # 47 core features


_TON_PROTO_VALS = [
   "tcp", "udp", "icmp", "arp", "ospf", "rarp", "ipv6-icmp", "ip",
   "igmp", "gre", "esp", "ah", "ipv6", "sctp", "other",
]
_TON_CONNSTATE_VALS = [
   "S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0",
   "RSTRH", "SH", "SHR", "OTH",
]
_TON_SERVICE_VALS = [
   "http", "ftp", "ftp-data", "smtp", "dns", "ssh", "ssl", "dhcp",
   "irc", "pop3", "imap", "rdp", "snmp", "smb", "other", "-",
]
_TON_DNSQTYPE_VALS = list(range(0, 66))
_TON_HTTP_METHOD_VALS = [
   "GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS",
   "PATCH", "CONNECT", "TRACE", "other",
]
_TON_HTTP_STATUS_VALS = [str(c) for c in [
   100, 101, 200, 201, 202, 204, 206, 301, 302, 304, 307, 308,
   400, 401, 403, 404, 405, 408, 409, 410, 413, 415, 422, 429,
   500, 501, 502, 503, 504,
]]
_TON_SSLVER_VALS = ["SSLv3", "TLSv10", "TLSv11", "TLSv12", "TLSv13", "other"]




def _expand_ton_cols() -> list:
   """
   Build the full ~470-column list matching the training pipeline's
   pd.get_dummies() output for ToN-IoT-v3.
   Called once at import time; result cached as _TON_COLS.
   """
   cols = list(_TON_COLS_CORE)
   for v in _TON_PROTO_VALS:       cols.append(f"proto_{v}")
   for v in _TON_CONNSTATE_VALS:   cols.append(f"conn_state_{v}")
   for v in _TON_SERVICE_VALS:     cols.append(f"service_{v}")
   for v in _TON_DNSQTYPE_VALS:    cols.append(f"dns_qtype_{v}")
   for v in _TON_HTTP_METHOD_VALS: cols.append(f"http_method_{v}")
   for v in _TON_HTTP_STATUS_VALS: cols.append(f"http_status_{v}")
   for v in _TON_SSLVER_VALS:      cols.append(f"ssl_version_{v}")
   while len(cols) < 470:
       cols.append(f"ton_feat_{len(cols)}")
   return cols




_TON_COLS = _expand_ton_cols()


DATASET_COLS = {
   "CIC-IDS-2017": _CIC2017_COLS,
   "CIC-IDS-2018": _CIC2018_COLS,
   "ToN-IoT-v3":   _TON_COLS,
   "Cross-Domain": _CIC2017_COLS,  # cross-domain models trained on CIC layout
}


# ---------------------------------------------------------------------------
# Model / scaler loaders
# ---------------------------------------------------------------------------
def load_framework_assets(dataset_dir: str, model_filename: str):
   """
   Load model, scaler, and target_names from dataset_dir.
   Returns (model, scaler, target_names); each is None if the file is absent.
   NOTE: Call this inside @st.cache_resource in app.py to avoid reloading.
   """
   model_path  = os.path.join(dataset_dir, model_filename)
   scaler_path = os.path.join(dataset_dir, "scaler.pkl")
   target_path = os.path.join(dataset_dir, "target_names.pkl")
   model = scaler = target_names = None


   if os.path.exists(model_path):
       if model_filename.endswith(".h5"):
           import tensorflow as tf
           model = tf.keras.models.load_model(model_path, compile=False)
       elif model_filename.endswith(".pkl"):
           model = joblib.load(model_path)


   if os.path.exists(scaler_path):
       scaler = joblib.load(scaler_path)
   if os.path.exists(target_path):
       target_names = joblib.load(target_path)


   return model, scaler, target_names




def load_csv_source(csv_path: str):
   """Load live_traffic_sample.csv; returns DataFrame or None."""
   if os.path.exists(csv_path):
       return pd.read_csv(csv_path)
   return None




def get_feature_count(active_scaler, active_model, dataset_choice: str) -> int:
   """
   Dynamically infer the required feature count from loaded assets.
   Priority: scaler.n_features_in_ -> model.input_shape -> dataset default.
   """
   if active_scaler and hasattr(active_scaler, "n_features_in_"):
       return int(active_scaler.n_features_in_)
   if active_model is not None:
       try:
           shape = active_model.input_shape
           if isinstance(shape, list):
               shape = shape[0]
           return int(shape[-1])
       except Exception:
           pass
   _defaults = {
       "CIC-IDS-2017": len(_CIC2017_COLS),   # 78
       "CIC-IDS-2018": len(_CIC2018_COLS),   # 80
       "ToN-IoT-v3":   len(_TON_COLS),        # ~470
       "Cross-Domain": len(_CIC2017_COLS),   # 78 (CIC layout)
   }
   return _defaults.get(dataset_choice, len(_TON_COLS))




# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def clean_label(label_str: str) -> str:
   """Normalise a raw class label to a clean human-readable string."""
   s = re.sub(r'[^\x00-\x7F]+', '-', str(label_str))
   s = s.replace('--', '-').strip('-').strip()
   if s.upper() in {"0", "0.0", "NORMAL", "BENIGN"}:
       return "Normal"
   if s in {"1", "1.0"}:
       return "Attack"
   return s if s else "Unknown"




def safe_int(val, default: int = 0) -> int:
   try:
       return int(float(val))
   except Exception:
       return default


def resolve_dataset_dir(monitoring_mode: str, balancing_choice: str,
                       dataset_choice: str) -> str:
   """Return the absolute path to the model/scaler/csv directory."""
   if monitoring_mode == "Cross-Domain (Robustness Test)":
       return os.path.join(BASE_DIR, "Cross_Validation")
   bal = BALANCING_FOLDER_MAP.get(balancing_choice, "")
   ds  = DATASET_FOLDER_MAP.get(dataset_choice, "")
   return os.path.join(BASE_DIR, bal, ds)


def resolve_model_file(monitoring_mode: str, model_choice: str) -> str:
   """Return the model filename for the given mode + model selection."""
   if monitoring_mode == "Cross-Domain (Robustness Test)" and model_choice == "CNN+Transformer":
       return "CNN_Transformer_model.h5"
   return MODEL_FILE_MAP.get(model_choice, "")