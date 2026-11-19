# 🚢 NexusChain AI — Real-Time Supply Chain Intelligence & Disruption Prediction

An enterprise-grade **Real-Time Supply Chain Intelligence & Disruption Prediction Platform** that continuously monitors global and regional logistics corridors, predicts shipment delays before they occur, forecasts multi-horizon product demand, estimates warehouse stockout probabilities, identifies high-risk suppliers, detects unusual shipping patterns, and recommends prescriptive alternative routes with full financial impact quantification.

---

## 🎯 Real-Time Scenario Demonstrated Out-of-the-Box

> **Live Telemetry Input:** Shipment **`SH-1042`** (Edge AI Neural Compute Modules, 3,000 units) is scheduled to reach **Mumbai (JNPT)** in **8 hours**. Live AIS and port telemetry detect an acute port congestion surge (`port_congestion_index = 0.89`).
>
> **Automated Platform Response:**
> *"Shipment SH-1042 has a 78% probability of arriving late. Expected delay: 11 hours. The primary cause is a port congestion signal. Recommended action: consider rerouting through an alternative logistics hub."*

---

## 🏗️ Architecture & Tech Stack

| Layer | Technologies Implemented | Module Path |
| :--- | :--- | :--- |
| **Delay & ETA Ensemble ML** | **XGBoost**, **LightGBM**, **CatBoost** (Classification + Regression + SHAP Root-Cause Attribution) | `src/ml/delay_predictor.py` |
| **Demand Forecasting** | **Temporal Fusion Transformer (TFT)** (Quantile P10/P50/P90 + Variable Selection + Attention) & **Prophet** | `src/ml/demand_forecaster.py` |
| **Inventory & Stockout Risk** | Coupled Inbound Delay + Demand Volatility Stockout Engine | `src/ml/stockout_predictor.py` |
| **Supplier Risk & Anomalies** | 5-Axis Multi-Tier Supplier Risk Scorer & **Isolation Forest** Kinematic Anomaly Detector | `src/ml/supplier_risk.py`, `src/ml/anomaly_detector.py` |
| **Prescriptive & Financial Engine** | Intermodal Rerouting Optimizer & Value-at-Risk / ROI Calculator | `src/ml/recommendation_engine.py` |
| **Real-Time Streaming** | **Apache Kafka** Producer/Consumer + **PySpark Structured Streaming** Windowed Aggregator | `src/streaming/kafka_pipeline.py`, `src/streaming/spark_streaming.py` |
| **Data & Low-Latency Cache** | **PostgreSQL** (`SQLAlchemy` with zero-config SQLite fallback) + **Redis** (with in-memory TTL fallback) | `src/data/database.py` |
| **Backend API** | **FastAPI** + Pydantic v2 REST & Telemetry Ingestion Service | `src/api/main.py` |
| **Interactive Command Center** | **Streamlit** + **Plotly** 7-Tab Executive Control Tower | `dashboard/app.py` |
| **MLOps & Drift Monitoring** | **MLflow** Experiment/Model Registry + **Evidently AI** Statistical Drift Suite (KS-Test & PSI) | `src/mlops/mlflow_tracker.py`, `src/mlops/drift_monitor.py` |

---

## 🚀 Quick Start

### 1. Run the End-to-End CLI Intelligence Demo
```bash
python run_demo.py
```

### 2. Launch the Interactive Streamlit + Plotly Executive Dashboard
```bash
streamlit run dashboard/app.py
```

### 3. Launch the FastAPI Backend Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI docs are available at `http://localhost:8000/docs`.

### 4. Run Automated Tests
```bash
pytest -v
```

### 5. Full Containerized Stack (PostgreSQL + Redis + Kafka + MLflow + FastAPI + Streamlit)
```bash
docker-compose up --build
```
