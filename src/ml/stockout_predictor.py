"""
Warehouse Stockout Probability & Monte Carlo Inventory Depletion Engine
Couples live inbound shipment delay predictions with daily demand volatility
to estimate stockout probabilities for individual warehouses.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List
import numpy as np

from src.data.database import repository
from src.ml.delay_predictor import delay_predictor
# overall purpose: Predict stockout probabilities for warehouses based on current inventory, inbound shipments, and demand patterns

class WarehouseStockoutPredictor:
    """
    Predicts stockout probability over 7-day and 14-day windows for each warehouse,
    incorporating live inbound shipment delays, safety stock thresholds, and demand variance.
    """

    def evaluate_warehouses(self) -> List[Dict[str, Any]]:
        warehouses = repository.get_warehouses()
        shipments_map = {s["shipment_id"]: s for s in repository.list_shipments()}
        results: List[Dict[str, Any]] = []

        for wh in warehouses:
            inbound_id = wh["inbound_shipment_id"]
            live_ship = shipments_map.get(inbound_id)
            if live_ship:
                pred = delay_predictor.predict_shipment(live_ship)
                expected_delay_hrs = pred["expected_delay_hours"]
                delay_prob = pred["delay_probability"]
            else:
                expected_delay_hrs = float(wh.get("inbound_delay_hours", 0.0))
                delay_prob = 0.5

            current_stock = float(wh["current_stock_units"])
            daily_mean = float(wh["daily_demand_mean"])
            daily_std = float(wh["daily_demand_std"])
            safety_stock = float(wh["safety_stock_units"])
            lead_time_days = float(wh["lead_time_days"])

            days_of_supply = current_stock / max(daily_mean, 1.0)
            effective_replenishment_days = (
                float(wh["inbound_eta_hours"]) + expected_delay_hrs
            ) / 24.0 + (lead_time_days * 0.45)

            # Expected demand during effective replenishment window
            expected_consumption = daily_mean * effective_replenishment_days
            consumption_std = daily_std * math.sqrt(max(effective_replenishment_days, 1.0))

            # Z-score for depleting current usable stock before replenishment lands
            z_depletion = (expected_consumption - (current_stock - safety_stock * 0.35)) / max(
                consumption_std, 1.0
            )
            base_stockout_prob = 1.0 / (1.0 + math.exp(-1.35 * z_depletion))
            # Amplify by inbound shipment delay risk
            stockout_prob_7d = min(
                0.98,
                max(0.02, 0.65 * base_stockout_prob + 0.35 * delay_prob * (1.0 if days_of_supply < 7.5 else 0.4)),
            )
            stockout_prob_14d = min(0.99, stockout_prob_7d * 1.22)

            # Simulate 10-day inventory trajectory (with vs without delay)
            trajectory = []
            inv_nominal = current_stock
            inv_delayed = current_stock
            eta_day_nominal = max(1, int(round(float(wh["inbound_eta_hours"]) / 24.0)))
            eta_day_delayed = max(
                eta_day_nominal + 1,
                int(round((float(wh["inbound_eta_hours"]) + expected_delay_hrs) / 24.0)) + 2,
            )

            for day in range(1, 11):
                inv_nominal = max(0.0, inv_nominal - daily_mean)
                inv_delayed = max(0.0, inv_delayed - daily_mean)
                if day == eta_day_nominal:
                    inv_nominal += float(wh["inbound_units"])
                if day == eta_day_delayed:
                    inv_delayed += float(wh["inbound_units"])
                trajectory.append(
                    {
                        "day": f"Day +{day}",
                        "inventory_on_time": round(inv_nominal),
                        "inventory_with_delay": round(inv_delayed),
                        "safety_stock": round(safety_stock),
                    }
                )

            if stockout_prob_7d >= 0.65:
                risk_tier = "CRITICAL"
                action = (
                    f"Trigger emergency lateral transfer of {int(daily_mean * 3)} units & "
                    f"expedite inbound shipment {inbound_id}."
                )
            elif stockout_prob_7d >= 0.35:
                risk_tier = "HIGH"
                action = f"Release safety stock buffer and prioritize unloading for {inbound_id}."
            else:
                risk_tier = "LOW"
                action = "Inventory healthy; continue automated replenishment."

            expected_shortfall_units = max(
                0.0, (daily_mean * 7.0 - current_stock) * stockout_prob_7d
            )
            financial_exposure_usd = round(
                expected_shortfall_units * float(wh["unit_margin_usd"]), 2
            )

            results.append(
                {
                    **wh,
                    "days_of_supply": round(days_of_supply, 2),
                    "inbound_delay_probability": round(delay_prob, 4),
                    "inbound_expected_delay_hours": round(expected_delay_hrs, 2),
                    "stockout_probability_7d": round(stockout_prob_7d, 4),
                    "stockout_probability_7d_pct": int(round(stockout_prob_7d * 100)),
                    "stockout_probability_14d": round(stockout_prob_14d, 4),
                    "risk_tier": risk_tier,
                    "expected_shortfall_units": int(round(expected_shortfall_units)),
                    "financial_exposure_usd": financial_exposure_usd,
                    "recommended_inventory_action": action,
                    "inventory_trajectory": trajectory,
                }
            )

        results.sort(key=lambda r: r["stockout_probability_7d"], reverse=True)
        return results


stockout_predictor = WarehouseStockoutPredictor()
