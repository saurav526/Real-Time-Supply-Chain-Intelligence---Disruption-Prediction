"""
Multi-Tier Supplier Risk Intelligence & Failure Probability Scoring
Evaluates delivery reliability, quality defects (PPM), financial health,
geopolitical exposure, lead-time variability, and single-source vulnerability.
"""
from __future__ import annotations

from typing import Any, Dict, List
from src.data.database import repository


class SupplierRiskEngine:
    """
    Identifies high-risk suppliers and recommends pre-qualified backup suppliers.
    """

    def analyze_suppliers(self) -> List[Dict[str, Any]]:
        suppliers = repository.get_suppliers()
        supplier_lookup = {s["supplier_id"]: s for s in suppliers}
        analyzed: List[Dict[str, Any]] = []

        for sup in suppliers:
            late_risk = 1.0 - float(sup["on_time_rate"])
            quality_risk = min(1.0, float(sup["defect_rate_ppm"]) / 2000.0)
            financial_risk = 1.0 - float(sup["financial_stability_score"])
            geo_risk = float(sup["geopolitical_risk_index"])
            lead_var_risk = min(
                1.0, float(sup["lead_time_std_days"]) / max(float(sup["lead_time_days_avg"]), 1.0) * 2.5
            )
            single_source_penalty = 0.12 if sup.get("single_source") else 0.0

            composite_score = min(
                0.98,
                (
                    0.28 * late_risk
                    + 0.18 * quality_risk
                    + 0.20 * financial_risk
                    + 0.22 * geo_risk
                    + 0.12 * lead_var_risk
                    + single_source_penalty
                ),
            )

            if composite_score >= 0.62:
                risk_level = "HIGH_RISK"
            elif composite_score >= 0.40:
                risk_level = "MODERATE_RISK"
            else:
                risk_level = "LOW_RISK"

            backup_id = sup.get("backup_supplier_id")
            backup_sup = supplier_lookup.get(backup_id, {})

            expected_annual_loss_usd = round(
                float(sup["annual_spend_usd"]) * composite_score * 0.14, 2
            )

            analyzed.append(
                {
                    **sup,
                    "composite_risk_score": round(composite_score, 4),
                    "composite_risk_pct": int(round(composite_score * 100)),
                    "risk_level": risk_level,
                    "risk_dimensions": {
                        "Delivery Delay Risk": round(late_risk, 3),
                        "Quality Defect Risk": round(quality_risk, 3),
                        "Financial Instability": round(financial_risk, 3),
                        "Geopolitical Exposure": round(geo_risk, 3),
                        "Lead-Time Volatility": round(lead_var_risk, 3),
                    },
                    "backup_supplier_name": backup_sup.get("name", "Dual-Source Pool"),
                    "backup_supplier_country": backup_sup.get("country", "India"),
                    "backup_lead_time_days": backup_sup.get("lead_time_days_avg", 10.0),
                    "value_at_risk_usd": expected_annual_loss_usd,
                    "mitigation_recommendation": (
                        f"Shift 35% allocation to {backup_sup.get('name', backup_id)} "
                        f"({backup_sup.get('country', '')}) to reduce lead-time by "
                        f"{max(0.0, round(float(sup['lead_time_days_avg']) - float(backup_sup.get('lead_time_days_avg', 10.0)), 1))} days."
                        if risk_level != "LOW_RISK"
                        else "Maintain strategic preferred-vendor status."
                    ),
                }
            )

        analyzed.sort(key=lambda s: s["composite_risk_score"], reverse=True)
        return analyzed


supplier_risk_engine = SupplierRiskEngine()
