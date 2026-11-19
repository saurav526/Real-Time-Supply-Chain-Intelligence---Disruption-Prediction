"""
Interactive CLI Demo & Verification Runner for NexusChain AI
Executes the entire ML, Forecasting, Streaming, Prescriptive, and MLOps pipeline
and prints executive intelligence summaries to the console.
"""
from __future__ import annotations

from src.data.database import repository
from src.ml.anomaly_detector import anomaly_detector
from src.ml.delay_predictor import delay_predictor
from src.ml.demand_forecaster import demand_forecaster
from src.ml.recommendation_engine import recommendation_engine
from src.ml.stockout_predictor import stockout_predictor
from src.ml.supplier_risk import supplier_risk_engine
from src.mlops.drift_monitor import drift_monitor
from src.mlops.mlflow_tracker import mlflow_registry


def main() -> None:
    repository.reset_shipments_to_default()
    print("=" * 88)
    print("  NEXUSCHAIN AI - REAL-TIME SUPPLY CHAIN INTELLIGENCE & DISRUPTION PREDICTION")
    print("=" * 88)

    # 1. Real-Time Shipment SH-1042 Scenario
    sh1042 = repository.get_shipment("SH-1042")
    pred_1042 = delay_predictor.predict_shipment(sh1042)
    print("\n[1] REAL-TIME TELEMETRY & DELAY PREDICTION (MUMBAI CORRIDOR SH-1042)")
    print(f"    Origin -> Destination : {sh1042['origin_port']} -> {sh1042['destination_port']}")
    print(f"    Scheduled ETA         : {sh1042['scheduled_eta_hours']} hours")
    print(f"    Ensemble Breakdown    : {pred_1042['model_breakdown']}")
    print(f"    >>> SYSTEM RESPONSE   : {pred_1042['alert_message']}")

    # 2. Prescriptive Route & Financial Impact
    rec_1042 = recommendation_engine.evaluate_shipment_disruption("SH-1042")
    fin = rec_1042["financial_impact"]
    best_opt = rec_1042["recommended_route_option"]
    print("\n[2] FINANCIAL IMPACT & PRESCRIPTIVE REROUTING (SH-1042)")
    print(f"    Cargo Value           : ${fin['cargo_value_usd']:,.2f}")
    print(f"    Unmitigated Loss      : ${fin['total_unmitigated_disruption_cost_usd']:,.2f}")
    print(f"    Optimal Alternative   : {best_opt['hub_name']} ({best_opt['mode']})")
    print(f"    Transit Time Saved    : {best_opt['transit_time_saved_hours']} hours")
    print(f"    Net Financial Benefit : ${fin['estimated_net_savings_usd']:,.2f} (ROI: {best_opt['roi_pct']}%)")

    # 3. Demand Forecasting (TFT + Prophet)
    fc = demand_forecaster.forecast_sku("SKU-SOC-900", horizon_days=14)
    print("\n[3] 14-DAY DEMAND FORECASTING (TEMPORAL FUSION TRANSFORMER + PROPHET)")
    print(f"    SKU                   : {fc['sku_id']}")
    print(f"    14-Day P50 Forecast   : {fc['total_predicted_demand_p50']:,.1f} units")
    print(f"    14-Day P90 Upper Bound: {fc['total_predicted_demand_p90']:,.1f} units")
    print(f"    Accuracy (TFT MAPE)   : {fc['metrics']['tft_mape_pct']}%")

    # 4. Warehouse Stockout Risk
    warehouses = stockout_predictor.evaluate_warehouses()
    print("\n[4] WAREHOUSE STOCKOUT PROBABILITY (TOP 3 AT-RISK HUBS)")
    for w in warehouses[:3]:
        print(
            f"    - {w['warehouse_id']} ({w['city']}): 7d Stockout Prob = {w['stockout_probability_7d_pct']}% "
            f"| Days of Supply = {w['days_of_supply']}d | Exposure = ${w['financial_exposure_usd']:,.0f}"
        )

    # 5. High-Risk Suppliers & Anomalies
    suppliers = supplier_risk_engine.analyze_suppliers()
    anomalies = anomaly_detector.detect_anomalies()
    print("\n[5] HIGH-RISK SUPPLIERS & ISOLATION FOREST SHIPPING ANOMALIES")
    for s in suppliers[:2]:
        print(
            f"    - Supplier {s['supplier_id']} ({s['name']}): Risk = {s['composite_risk_pct']}% ({s['risk_level']}) "
            f"-> Backup: {s['backup_supplier_name']}"
        )
    for a in [x for x in anomalies if x["is_anomaly"]][:2]:
        print(
            f"    - Shipment {a['shipment_id']} Anomaly Score = {a['anomaly_score']} ({a['anomaly_severity']}): "
            f"{', '.join(a['detected_patterns'])}"
        )

    # 6. MLOps & Drift Monitoring
    reg = mlflow_registry.get_registry_overview()
    drift = drift_monitor.evaluate_drift()
    print("\n[6] MLOPS REGISTRY (MLFLOW) & DATA DRIFT MONITOR (EVIDENTLY AI)")
    for m in reg["registered_models"]:
        print(
            f"    - {m['model_name']:<30} | ROC-AUC: {m['roc_auc']:.4f} | F1: {m['f1_score']:.4f} | MAE: {m['mae_hours']:.2f}h"
        )
    print(
        f"    - Evidently Drift Status: {drift['drift_summary_status']} "
        f"({drift['drifted_features_count']}/10 features shifted)"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
