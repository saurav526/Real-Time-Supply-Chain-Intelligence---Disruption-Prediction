"""
Unusual Shipping Pattern & Telemetry Anomaly Detector
Uses Isolation Forest trained on historical vessel/freight kinematics
plus domain rules for route deviations, AIS dark periods, speed drops, and customs dwell spikes.
"""
from __future__ import annotations
# Main purpose: Detect unusual shipping patterns across live shipments using IsolationForest and multi-signal kinematic heuristics.
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data.database import repository
from src.data.generator import generate_historical_shipments

ANOMALY_FEATURES = [
    "vessel_speed_knots",
    "route_deviation_km",
    "ais_signal_dropouts",
    "port_congestion_index",
    "customs_hold_prob",
]


class ShippingAnomalyDetector:
    """
    Detects unusual shipping patterns across live shipments using IsolationForest
    and multi-signal kinematic heuristics.
    """

    def __init__(self) -> None:
        hist_df = generate_historical_shipments(n_samples=1500, seed=42)
        self.model = IsolationForest(
            n_estimators=150, contamination=0.12, random_state=42
        )
        self.model.fit(hist_df[ANOMALY_FEATURES])

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        shipments = repository.list_shipments()
        if not shipments:
            return []

        df = pd.DataFrame(shipments)
        X = df[ANOMALY_FEATURES].astype(float)
        raw_preds = self.model.predict(X)  # -1 = anomaly, 1 = normal
        raw_scores = -self.model.score_samples(X)  # higher = more anomalous

        results: List[Dict[str, Any]] = []
        for idx, ship in enumerate(shipments):
            score_norm = float(np.clip((raw_scores[idx] - 0.38) / 0.35, 0.05, 0.99))
            patterns: List[str] = []

            if float(ship["route_deviation_km"]) > 80.0:
                patterns.append(
                    f"Off-corridor route deviation of {ship['route_deviation_km']:.0f} km"
                )
            if int(ship["ais_signal_dropouts"]) >= 3:
                patterns.append(
                    f"Unusual AIS transponder dark periods ({ship['ais_signal_dropouts']} dropouts)"
                )
            if float(ship["vessel_speed_knots"]) < 12.5:
                patterns.append(
                    f"Abnormal vessel speed drop ({ship['vessel_speed_knots']} knots vs 17.5 kt norm)"
                )
            if float(ship["port_congestion_index"]) > 0.75:
                patterns.append(
                    f"Extreme destination port congestion surge ({int(float(ship['port_congestion_index'])*100)}%)"
                )
            if float(ship["customs_hold_prob"]) > 0.60:
                patterns.append(
                    f"Anomalous customs inspection dwell signal ({int(float(ship['customs_hold_prob'])*100)}%)"
                )

            is_anomalous = bool(raw_preds[idx] == -1 or len(patterns) >= 2)
            if is_anomalous and score_norm < 0.62:
                score_norm = min(0.95, 0.64 + 0.08 * len(patterns))

            results.append(
                {
                    "shipment_id": ship["shipment_id"],
                    "origin_port": ship["origin_port"],
                    "destination_port": ship["destination_port"],
                    "carrier": ship["carrier"],
                    "is_anomaly": is_anomalous,
                    "anomaly_score": round(score_norm, 3),
                    "anomaly_severity": (
                        "HIGH" if score_norm >= 0.72 else ("MEDIUM" if is_anomalous else "NORMAL")
                    ),
                    "detected_patterns": patterns if patterns else ["Normal corridor kinematics"],
                    "vessel_speed_knots": ship["vessel_speed_knots"],
                    "route_deviation_km": ship["route_deviation_km"],
                    "ais_signal_dropouts": ship["ais_signal_dropouts"],
                }
            )

        results.sort(key=lambda r: r["anomaly_score"], reverse=True)
        return results


anomaly_detector = ShippingAnomalyDetector()
