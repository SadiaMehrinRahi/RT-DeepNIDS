---
title: RT-DeepNIDS
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.50.0
python_version: "3.11"
app_file: dashboard.py
pinned: false
---

# RT-DeepNIDS Live Demo

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Space-blue?logo=huggingface)](https://sadiamehrinrahi-rt-deepnids.hf.space/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-gray)

Online deployment of **RT-DeepNIDS: A Real-Time Hybrid Network Intrusion
Detection System for IT and IoT Environments** the hosted, browser-based
dashboard that serves the trained models for interactive intrusion detection.

> Undergraduate capstone research — Department of Computer Science and
> Engineering, Bangladesh University of Business and Technology (BUBT),
> Dhaka, Bangladesh.

---

## Live Demo

**https://sadiamehrinrahi-rt-deepnids.hf.space/**

Deployed on **Hugging Face Spaces**. Just open the link, pick a configuration in
the control panel, and press **Start** no installation needed.

---

## About This Version

This repository is the **online (cloud) build** of RT-DeepNIDS. It runs in
**CSV Simulation** mode only: cloud hosting cannot capture live network
packets, so the dashboard replays exported test traffic through the trained
models. The full version with **live Scapy packet capture** runs locally
see the [main project repository](https://github.com/sanjida-khanom/RT-DeepNIDS-A-Real-Time-Hybrid-Network-Intrusion-Detection-System-for-IT-and-IoT-Environments)
and its [`Real_Time_Deployment/`](https://github.com/sanjida-khanom/RT-DeepNIDS-A-Real-Time-Hybrid-Network-Intrusion-Detection-System-for-IT-and-IoT-Environments/tree/main/Real_Time_Deployment)
folder.

| Capability | This online build |
|------------|:------------------:|
| CSV Simulation (replay test traffic) | ✅ |
| Live packet capture (Scapy) | ❌ (local only) |
| All ML / DL models | ✅ |
| SHAP explanations | ✅ |
| Port-scan heuristic | ✅ |

> **Note:** Because of the hosting storage limit, not every trained model is
> uploaded to the Space (the **ToN-IoT-v3 · Random Forest** file alone is over
> 1 GB). Every other configuration runs normally online, and the **complete
> `Real_Time_Export/` model set** can be downloaded here:
>
> **Models & assets:** https://tinyurl.com/Real-Time-Export

---

## Trained Models

The complete set of trained models, scalers, and sample traffic files
(`Real_Time_Export/`) is hosted on Google Drive (too large for GitHub /
Hugging Face):

**Download: https://tinyurl.com/Real-Time-Export**

After downloading, place the `Real_Time_Export/` folder in the project root so
it sits next to `dashboard.py`, then run the app.

---

## Full Project & Local Deployment

- **Full project (training pipelines, datasets, methodology, deployment):**
  [Capstone-Project/RT-DeepNIDS](https://github.com/sanjida-khanom/RT-DeepNIDS-A-Real-Time-Hybrid-Network-Intrusion-Detection-System-for-IT-and-IoT-Environments)

- **Local deployment with live Scapy packet capture** full setup and usage
  instructions are in the
  [`Real_Time_Deployment/`](https://github.com/sanjida-khanom/RT-DeepNIDS-A-Real-Time-Hybrid-Network-Intrusion-Detection-System-for-IT-and-IoT-Environments/tree/main/Real_Time_Deployment)
  folder of the main repository.

---

## Features

- **CSV Simulation:** Streams exported test traffic through the trained models
  in real time, matching the reported evaluation accuracy.
- **Models:** Decision Tree, Random Forest, XGBoost, Hybrid CNN-GRU, and
  CNN+Transformer.
- **Four evaluation modes:** CIC-IDS-2017, CIC-IDS-2018, ToN-IoT-v3, and
  Cross-Domain (zero-day robustness test).
- **Dual-pipeline:** SMOTE (high precision) vs. Tomek Links + IHT (high speed).
- **Explainable AI (SHAP):** Every attack verdict can be explained on demand.
- **Live KPIs & charts:** Threat classification, traffic activity, and a
  streaming detection log.

---

## Project Structure

```
.
├── dashboard.py          # Streamlit app (UI, dashboard, SHAP panel)
├── engine.py             # Model / scaler loaders and feature templates
├── requirements.txt      # Python dependencies
├── .devcontainer/        # Dev container config
└── Real_Time_Export/     # Trained models, scalers & sample traffic
    ├── SMOTE/  ├── Tomek_IHT/  └── Cross_Validation/
```

---

## Run Locally (Optional)
 
Pick the version you need:
 
### Option A: CSV Simulation only (this repo)
 
This build runs **CSV Simulation** only (same as the hosted demo).
 
```bash
# 1. Clone
git clone https://github.com/SadiaMehrinRahi/RT-DeepNIDS.git
cd RT-DeepNIDS
 
# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux / macOS
 
# 3. Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
 
# 4. Run
streamlit run dashboard.py
```
The dashboard opens at `http://localhost:8501`.
 
### Option B: CSV Simulation **and** Live Network capture
 
For **both** CSV Simulation and live network (Wi-Fi / Ethernet) packet capture,
use the `Real_Time_Deployment/` folder in the main project repository, it has
the full app and step-by-step setup instructions:
 
**https://github.com/sanjida-khanom/RT-DeepNIDS-A-Real-Time-Hybrid-Network-Intrusion-Detection-System-for-IT-and-IoT-Environments/tree/main/Real_Time_Deployment**
 
```bash
git clone https://github.com/sanjida-khanom/RT-DeepNIDS-A-Real-Time-Hybrid-Network-Intrusion-Detection-System-for-IT-and-IoT-Environments.git
cd RT-DeepNIDS-A-Real-Time-Hybrid-Network-Intrusion-Detection-System-for-IT-and-IoT-Environments/Real_Time_Deployment
# then follow that folder's README (needs admin/sudo + Npcap for live capture)
```

---

## Deployment (Hugging Face Spaces)

This repo is configured for **Hugging Face Spaces** via the YAML block at the
top of this README (`sdk: streamlit`, `python_version: 3.11`,
`app_file: dashboard.py`). Pushing to the Space rebuilds and redeploys it
automatically.

---

## Team

**Supervisor:** Md. Saifur Rahman, Assistant Professor, Dept. of CSE, BUBT

| Member | ID |
|--------|-----|
| Ayesha Siddika | 22234103099 |
| Sanjida Khanom | 22234103103 |
| Ihsanul Hossain Rafsan | 22234103112 |
| Sadia Mehrin Rahi | 22234103122 |
| Istiyak Hasan Maruf | 22234103130 |

Bangladesh University of Business and Technology (BUBT) — Department of Computer
Science and Engineering.
