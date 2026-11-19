# FastAPI Backend Service — Real-Time Supply Chain Intelligence & Disruption Prediction
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.data.database import repository
from src.ml.anomaly_detector import anomaly_detector
from src.ml.delay_predictor import delay_predictor
from src.ml.demand_forecaster import demand_forecaster
from src.ml.recommendation_engine import recommendation_engine
from src.ml.stockout_predictor import stockout_predictor
from src.ml.supplier_risk import supplier_risk_engine
from src.mlops.drift_monitor import drift_monitor
from src.mlops.mlflow_tracker import mlflow_registry
from src.streaming.kafka_pipeline import kafka_pipeline
from src.streaming.spark_streaming import spark_processor

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Real-Time Supply Chain Intelligence & Disruption Prediction Platform "
        "powered by XGBoost, LightGBM, CatBoost, Temporal Fusion Transformer (TFT), "
        "Prophet, Apache Kafka, PySpark, PostgreSQL, Redis, MLflow, and Evidently AI."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TelemetryUpdatePayload(BaseModel):
    port_congestion_index: Optional[float] = Field(None, ge=0.0, le=1.0)
    weather_severity: Optional[float] = Field(None, ge=0.0, le=1.0)
    customs_hold_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    vessel_speed_knots: Optional[float] = Field(None, ge=2.0, le=30.0)
    route_deviation_km: Optional[float] = Field(None, ge=0.0, le=1000.0)
    ais_signal_dropouts: Optional[int] = Field(None, ge=0, le=50)
    scheduled_eta_hours: Optional[float] = Field(None, ge=0.5, le=240.0)


@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "operational",
        "service": settings.app_name,
        "version": settings.version,
        "storage_backend": repository.db_backend_name,
        "cache_backend": repository.cache.backend_name,
        "streaming_broker": kafka_pipeline.broker_mode,
        "mlflow_tracking": mlflow_registry.tracking_uri,
        "active_models": list(delay_predictor.model_metrics.keys()),
    }


@app.get("/api/v1/shipments")
async def list_shipments_with_predictions() -> Dict[str, Any]:
    shipments = repository.list_shipments()
    enriched: List[Dict[str, Any]] = []
    for s in shipments:
        pred = delay_predictor.predict_shipment(s)
        enriched.append({**s, "prediction": pred})
    return {
        "count": len(enriched),
        "shipments": enriched,
    }


@app.get("/api/v1/shipments/{shipment_id}/predict")
async def predict_single_shipment(shipment_id: str) -> Dict[str, Any]:
    shipment = repository.get_shipment(shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail=f"Shipment '{shipment_id}' not found")
    prediction = delay_predictor.predict_shipment(shipment)
    return {
        "shipment": shipment,
        "prediction": prediction,
    }


@app.post("/api/v1/shipments/{shipment_id}/telemetry")
async def ingest_live_telemetry(shipment_id: str, payload: TelemetryUpdatePayload) -> Dict[str, Any]:
    patch = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not patch:
        raise HTTPException(status_code=400, detail="No telemetry fields provided")
    try:
        event = kafka_pipeline.publish_telemetry_update(shipment_id, patch)
        return event
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/shipments/reset")
async def reset_shipments() -> Dict[str, Any]:
    shipments = repository.reset_shipments_to_default()
    return {"status": "reset_complete", "count": len(shipments)}


@app.get("/api/v1/forecast/{sku_id}")
async def get_demand_forecast(
    sku_id: str, horizon_days: int = Query(14, ge=7, le=30)
) -> Dict[str, Any]:
    return demand_forecaster.forecast_sku(sku_id=sku_id, horizon_days=horizon_days)


@app.get("/api/v1/warehouses/stockout")
async def get_warehouse_stockout_predictions() -> Dict[str, Any]:
    warehouses = stockout_predictor.evaluate_warehouses()
    return {
        "count": len(warehouses),
        "warehouses": warehouses,
    }


@app.get("/api/v1/suppliers/risk")
async def get_supplier_risk_profiles() -> Dict[str, Any]:
    suppliers = supplier_risk_engine.analyze_suppliers()
    return {
        "count": len(suppliers),
        "high_risk_count": sum(1 for s in suppliers if s["risk_level"] == "HIGH_RISK"),
        "suppliers": suppliers,
    }


@app.get("/api/v1/anomalies")
async def get_shipping_anomalies() -> Dict[str, Any]:
    anomalies = anomaly_detector.detect_anomalies()
    return {
        "count": len(anomalies),
        "anomalous_shipments_count": sum(1 for a in anomalies if a["is_anomaly"]),
        "shipments": anomalies,
    }


@app.get("/api/v1/recommendations/{shipment_id}")
async def get_shipment_recommendations(shipment_id: str) -> Dict[str, Any]:
    shipment = repository.get_shipment(shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail=f"Shipment '{shipment_id}' not found")
    return recommendation_engine.evaluate_shipment_disruption(shipment_id)


@app.get("/api/v1/financial-impact")
async def get_portfolio_financial_impact() -> Dict[str, Any]:
    return recommendation_engine.compute_portfolio_financial_summary()


@app.get("/api/v1/streaming/windowed-metrics")
async def get_streaming_metrics() -> Dict[str, Any]:
    return {
        "broker_mode": kafka_pipeline.broker_mode,
        "windowed_port_metrics": spark_processor.compute_windowed_port_metrics(),
        "recent_stream_events": kafka_pipeline.get_recent_stream_events(limit=15),
        "spark_structured_streaming_spec": spark_processor.build_spark_streaming_code_spec(),
    }


@app.get("/api/v1/mlops/overview")
async def get_mlops_overview() -> Dict[str, Any]:
    return {
        "mlflow_registry": mlflow_registry.get_registry_overview(),
        "evidently_drift": drift_monitor.evaluate_drift(),
    }
