"""
Prescriptive Rerouting, Alternative Supplier Selection & Financial Impact Estimator
Quantifies the financial cost of doing nothing vs. executing the AI-recommended
mitigation (rerouting through an alternative logistics hub or activating backup supplier).
"""
from __future__ import annotations

from typing import Any, Dict, List
from src.data.database import repository
from src.data.generator import PORTS_AND_HUBS
from src.ml.delay_predictor import delay_predictor


ROUTE_ALTERNATIVES_DB: Dict[str, List[Dict[str, Any]]] = {
    "Mumbai (JNPT)": [
        {
            "option_id": "ALT-MUM-01",
            "hub_name": "Mundra Port (INMUN) + Dedicated Western Freight Rail Corridor",
            "mode": "Sea-to-Rail Intermodal",
            "transit_time_saved_hours": 7.5,
            "residual_delay_hours": 3.5,
            "reroute_cost_usd": 14200.0,
            "congestion_index": 0.34,
            "reliability_score": 0.93,
            "waypoints": [
                {"lat": 17.62, "lon": 72.18, "label": "Current Vessel Position"},
                {"lat": 22.7381, "lon": 69.7041, "label": "Mundra Port Hub"},
                {"lat": 19.2813, "lon": 73.0483, "label": "Mumbai Bhiwandi DC"},
            ],
        },
        {
            "option_id": "ALT-MUM-02",
            "hub_name": "Nhava Sheva Express Feeder + Pune Dry Port Customs Clearance",
            "mode": "Priority Barge + Express Highway",
            "transit_time_saved_hours": 6.0,
            "residual_delay_hours": 5.0,
            "reroute_cost_usd": 9800.0,
            "congestion_index": 0.41,
            "reliability_score": 0.89,
            "waypoints": [
                {"lat": 17.62, "lon": 72.18, "label": "Current Vessel Position"},
                {"lat": 18.5204, "lon": 73.8567, "label": "Pune Dry Port Hub"},
                {"lat": 19.2813, "lon": 73.0483, "label": "Mumbai Bhiwandi DC"},
            ],
        },
        {
            "option_id": "ALT-MUM-03",
            "hub_name": "Emergency Air-Bridge Charter (Critical Batch Split 25%)",
            "mode": "Air Freight Express",
            "transit_time_saved_hours": 9.5,
            "residual_delay_hours": 1.5,
            "reroute_cost_usd": 28500.0,
            "congestion_index": 0.12,
            "reliability_score": 0.98,
            "waypoints": [
                {"lat": 1.2644, "lon": 103.8222, "label": "Singapore Changi Air Hub"},
                {"lat": 19.0896, "lon": 72.8656, "label": "Mumbai CSIA Cargo Terminal"},
            ],
        },
    ],
    "Chennai": [
        {
            "option_id": "ALT-MAA-01",
            "hub_name": "Ennore Kamarajar Port (INENR) Fast-Track Berth",
            "mode": "Coastal Diversion + Express Truck",
            "transit_time_saved_hours": 11.0,
            "residual_delay_hours": 4.5,
            "reroute_cost_usd": 11500.0,
            "congestion_index": 0.29,
            "reliability_score": 0.91,
            "waypoints": [
                {"lat": 9.85, "lon": 83.45, "label": "Current Vessel Position"},
                {"lat": 13.2625, "lon": 80.3269, "label": "Ennore Kamarajar Hub"},
                {"lat": 12.9716, "lon": 77.5946, "label": "Bengaluru / Chennai DC"},
            ],
        },
        {
            "option_id": "ALT-MAA-02",
            "hub_name": "Krishnapatnam Deepwater Terminal + NH-16 Corridor",
            "mode": "Ocean + Bonded Trucking",
            "transit_time_saved_hours": 8.5,
            "residual_delay_hours": 6.5,
            "reroute_cost_usd": 8900.0,
            "congestion_index": 0.33,
            "reliability_score": 0.88,
            "waypoints": [],
        },
    ],
}


class RecommendationAndFinancialEngine:
    """
    Computes the financial impact of a supply chain disruption and recommends
    optimal alternative shipping routes and backup suppliers.
    """

    def evaluate_shipment_disruption(self, shipment_id: str) -> Dict[str, Any]:
        ship = repository.get_shipment(shipment_id)
        if not ship:
            shipments = repository.list_shipments()
            ship = shipments[0]

        pred = delay_predictor.predict_shipment(ship)
        delay_hrs = float(pred["expected_delay_hours"])
        delay_prob = float(pred["delay_probability"])
        cargo_val = float(ship.get("cargo_value_usd", 500000.0))
        sla_rate = float(ship.get("sla_penalty_per_hour_usd", 2500.0))

        # Financial impact breakdown if no action is taken
        sla_penalty_cost = round(delay_hrs * sla_rate, 2)
        stockout_production_halt_cost = round(
            cargo_val * 0.085 * delay_prob * (delay_hrs / 10.0), 2
        )
        demurrage_detention_cost = round(max(0.0, (delay_hrs - 4.0) * 950.0), 2)
        spoiling_or_holding_cost = round(cargo_val * 0.004 * (delay_hrs / 8.0), 2)

        total_unmitigated_loss = round(
            sla_penalty_cost
            + stockout_production_halt_cost
            + demurrage_detention_cost
            + spoiling_or_holding_cost,
            2,
        )
        expected_risk_adjusted_loss = round(total_unmitigated_loss * delay_prob, 2)

        dest_port = ship.get("destination_port", "Mumbai (JNPT)")
        alt_routes = ROUTE_ALTERNATIVES_DB.get(dest_port)
        if not alt_routes:
            hub_meta = PORTS_AND_HUBS.get(dest_port, PORTS_AND_HUBS["Mumbai (JNPT)"])
            alt_name = hub_meta["alternative_hubs"][0]
            alt_routes = [
                {
                    "option_id": f"ALT-{shipment_id}-01",
                    "hub_name": f"{alt_name} Intermodal Priority Corridor",
                    "mode": "Sea-to-Land Express",
                    "transit_time_saved_hours": round(delay_hrs * 0.65, 1),
                    "residual_delay_hours": round(delay_hrs * 0.35, 1),
                    "reroute_cost_usd": 12400.0,
                    "congestion_index": 0.31,
                    "reliability_score": 0.92,
                    "waypoints": [],
                }
            ]

        enriched_options: List[Dict[str, Any]] = []
        for opt in alt_routes:
            saved_ratio = min(
                0.90,
                float(opt["transit_time_saved_hours"]) / max(delay_hrs, 1.0),
            )
            avoided_loss = round(total_unmitigated_loss * saved_ratio, 2)
            net_savings = round(avoided_loss - float(opt["reroute_cost_usd"]), 2)
            roi_pct = round(
                (net_savings / max(float(opt["reroute_cost_usd"]), 1.0)) * 100.0, 1
            )
            enriched_options.append(
                {
                    **opt,
                    "avoided_disruption_loss_usd": avoided_loss,
                    "net_financial_benefit_usd": net_savings,
                    "roi_pct": roi_pct,
                }
            )

        enriched_options.sort(key=lambda o: o["net_financial_benefit_usd"], reverse=True)
        best_option = enriched_options[0]

        # Lookup supplier & backup supplier
        suppliers = {s["supplier_id"]: s for s in repository.get_suppliers()}
        cur_sup = suppliers.get(ship.get("supplier_id", "SUP-101"), {})
        backup_sup = suppliers.get(cur_sup.get("backup_supplier_id", "SUP-104"), {})

        return {
            "shipment_id": ship["shipment_id"],
            "destination_port": dest_port,
            "prediction_summary": pred,
            "financial_impact": {
                "cargo_value_usd": cargo_val,
                "sla_penalty_cost_usd": sla_penalty_cost,
                "stockout_and_line_stop_cost_usd": stockout_production_halt_cost,
                "demurrage_and_detention_usd": demurrage_detention_cost,
                "capital_holding_cost_usd": spoiling_or_holding_cost,
                "total_unmitigated_disruption_cost_usd": total_unmitigated_loss,
                "risk_adjusted_expected_loss_usd": expected_risk_adjusted_loss,
                "recommended_reroute_cost_usd": best_option["reroute_cost_usd"],
                "estimated_net_savings_usd": max(0.0, best_option["net_financial_benefit_usd"]),
            },
            "recommended_route_option": best_option,
            "alternative_route_options": enriched_options,
            "supplier_contingency": {
                "current_supplier_id": cur_sup.get("supplier_id", "SUP-101"),
                "current_supplier_name": cur_sup.get("name", "Primary Supplier"),
                "current_supplier_country": cur_sup.get("country", "Taiwan"),
                "backup_supplier_id": backup_sup.get("supplier_id", "SUP-104"),
                "backup_supplier_name": backup_sup.get("name", "Bharat Forge & Silicon Systems"),
                "backup_supplier_country": backup_sup.get("country", "India"),
                "backup_lead_time_days": backup_sup.get("lead_time_days_avg", 9.5),
            },
        }

    def compute_portfolio_financial_summary(self) -> Dict[str, Any]:
        shipments = repository.list_shipments()
        evaluations = [self.evaluate_shipment_disruption(s["shipment_id"]) for s in shipments]
        total_cargo = sum(e["financial_impact"]["cargo_value_usd"] for e in evaluations)
        total_unmitigated = sum(
            e["financial_impact"]["total_unmitigated_disruption_cost_usd"] for e in evaluations
        )
        total_expected_loss = sum(
            e["financial_impact"]["risk_adjusted_expected_loss_usd"] for e in evaluations
        )
        total_net_savings = sum(
            e["financial_impact"]["estimated_net_savings_usd"] for e in evaluations
        )
        return {
            "active_shipments_count": len(evaluations),
            "total_cargo_value_at_sea_usd": round(total_cargo, 2),
            "total_unmitigated_disruption_exposure_usd": round(total_unmitigated, 2),
            "total_risk_adjusted_var_usd": round(total_expected_loss, 2),
            "total_prescriptive_net_savings_usd": round(total_net_savings, 2),
            "shipment_evaluations": evaluations,
        }


recommendation_engine = RecommendationAndFinancialEngine()
