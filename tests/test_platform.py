"""
End-to-End Automated Test Suite for NexusChain AI
Verifies:
  1. Canonical Real-Time Scenario (Shipment SH-1042 to Mumbai: 78% delay prob, 11h delay, port congestion cause)
  2. Multi-Model Ensemble (XGBoost, LightGBM, CatBoost) training & inference
  3. Temporal Fusion Transformer (TFT) + Prophet multi-horizon demand forecasting
  4. Warehouse stockout probability & inventory trajectory prediction
  5. Supplier risk scoring & Isolation Forest shipping pattern anomaly detection
  6. Prescriptive route recommendation & financial impact calculation
  7. FastAPI REST & Kafka telemetry streaming endpoints
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
from src.api.main import app
from src.data.database import repository
from src.ml.anomaly_detector import anomaly_detector
from src.ml.delay_predictor import delay_predictor
from src.ml.demand_forecaster import demand_forecaster
from src.ml.recommendation_engine import recommendation_engine
from src.ml.stockout_predictor import stockout_predictor
from src.ml.supplier_risk import supplier_risk_engine
from src.mlops.drift_monitor import drift_monitor
from src.mlops.mlflow_tracker import mlflow_registry

def test_canonical_sh1042_mumbai_realtime_scenario() -> None:
    repository.reset_shipments_to_default()
    sh1042 = repository.get_shipment("SH-1042")
    assert sh1042 is not None
    assert sh1042["destination_port"] == "Mumbai (JNPT)"
    assert sh1042["scheduled_eta_hours"] == 8.0

    pred = delay_predictor.predict_shipment(sh1042)
    assert pred["delay_probability_pct"] == 78
    assert pred["expected_delay_hours"] == 11.0
    assert pred["primary_cause"] == "port congestion signal"
    assert (
        pred["alert_message"]
        == "Shipment SH-1042 has a 78% probability of arriving late. Expected delay: 11 hours. "
        "The primary cause is a port congestion signal. Recommended action: consider rerouting through an alternative logistics hub."
    )
    assert set(pred["model_breakdown"].keys()) == {"XGBoost", "LightGBM", "CatBoost"}


def test_demand_forecaster_tft_and_prophet() -> None:
    res = demand_forecaster.forecast_sku("SKU-SOC-900", horizon_days=14)
    assert len(res["forecast_series"]) == 14
    for day_pred in res["forecast_series"]:
        assert day_pred["tft_p10"] <= day_pred["tft_p50"] <= day_pred["tft_p90"]
        assert day_pred["prophet_forecast"] > 0


def test_warehouse_stockout_predictions() -> None:
    warehouses = stockout_predictor.evaluate_warehouses()
    assert len(warehouses) == 5
    mum_wh = next(w for w in warehouses if w["warehouse_id"] == "WH-MUM-01")
    assert 0.0 <= mum_wh["stockout_probability_7d"] <= 1.0
    assert len(mum_wh["inventory_trajectory"]) == 10


def test_supplier_risk_and_anomaly_detection() -> None:
    suppliers = supplier_risk_engine.analyze_suppliers()
    assert len(suppliers) >= 5
    assert any(s["risk_level"] == "HIGH_RISK" for s in suppliers)

    anomalies = anomaly_detector.detect_anomalies()
    assert len(anomalies) == 6
    assert any(a["is_anomaly"] for a in anomalies)


def test_recommendations_and_financial_impact() -> None:
    rec = recommendation_engine.evaluate_shipment_disruption("SH-1042")
    fin = rec["financial_impact"]
    assert fin["total_unmitigated_disruption_cost_usd"] > 0
    assert fin["estimated_net_savings_usd"] > 0
    assert len(rec["alternative_route_options"]) >= 2


def test_fastapi_endpoints_and_kafka_telemetry_stream() -> None:
    async def _run_api_checks() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            r_health = await ac.get("/api/v1/health")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "operational"

            r_pred = await ac.get("/api/v1/shipments/SH-1042/predict")
            assert r_pred.status_code == 200
            assert r_pred.json()["prediction"]["delay_probability_pct"] == 78

            # Simulate live telemetry update via Kafka pipeline endpoint
            r_tel = await ac.post(
                "/api/v1/shipments/SH-1042/telemetry",
                json={"port_congestion_index": 0.25, "vessel_speed_knots": 19.5},
            )
            assert r_tel.status_code == 200
            updated_prob = r_tel.json()["prediction"]["delay_probability"]
            assert updated_prob < 0.78

            # Reset back to baseline
            r_reset = await ac.post("/api/v1/shipments/reset")
            assert r_reset.status_code == 200

    asyncio.run(_run_api_checks())


def test_mlops_registry_and_evidently_drift() -> None:
    reg = mlflow_registry.get_registry_overview()
    assert len(reg["registered_models"]) == 4

    drift = drift_monitor.evaluate_drift()
    assert len(drift["feature_drift_reports"]) == 10
