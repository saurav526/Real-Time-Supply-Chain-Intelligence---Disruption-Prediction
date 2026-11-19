"""
Synthetic & Realistic Supply Chain Telemetry, Historical Shipments,
Port Congestion, Warehouse Inventory, and Supplier Data Generator.

Includes the canonical real-time scenario:
  Shipment SH-1042 bound for Mumbai (JNPT) expected in 8 hours ->
  78% probability of arriving late, expected delay: 11 hours,
  primary cause: port congestion signal, recommended action: reroute via alternative logistics hub.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


PORTS_AND_HUBS: Dict[str, Dict[str, Any]] = {
    "Mumbai (JNPT)": {
        "code": "INNSA",
        "lat": 18.9490,
        "lon": 72.9512,
        "country": "India",
        "base_congestion": 0.84,
        "avg_dwell_hours": 38.0,
        "alternative_hubs": ["Mundra Port (INMUN)", "Pune Dry Port / Air Express Hub", "Chennai Port (INMAA)"],
    },
    "Mundra": {
        "code": "INMUN",
        "lat": 22.7381,
        "lon": 69.7041,
        "country": "India",
        "base_congestion": 0.34,
        "avg_dwell_hours": 16.5,
        "alternative_hubs": ["Kandla (INIXY)", "Mumbai (JNPT)"],
    },
    "Chennai": {
        "code": "INMAA",
        "lat": 13.0827,
        "lon": 80.2707,
        "country": "India",
        "base_congestion": 0.42,
        "avg_dwell_hours": 21.0,
        "alternative_hubs": ["Ennore Kamarajar (INENR)", "Krishnapatnam"],
    },
    "Singapore": {
        "code": "SGSIN",
        "lat": 1.2644,
        "lon": 103.8222,
        "country": "Singapore",
        "base_congestion": 0.58,
        "avg_dwell_hours": 24.0,
        "alternative_hubs": ["Port Klang (MYPKG)", "Tanjung Pelepas (MYTPP)"],
    },
    "Shanghai": {
        "code": "CNSHA",
        "lat": 31.2304,
        "lon": 121.4737,
        "country": "China",
        "base_congestion": 0.71,
        "avg_dwell_hours": 31.0,
        "alternative_hubs": ["Ningbo-Zhoushan (CNNGB)", "Qingdao (CNTAO)"],
    },
    "Rotterdam": {
        "code": "NLRTM",
        "lat": 51.9244,
        "lon": 4.4777,
        "country": "Netherlands",
        "base_congestion": 0.49,
        "avg_dwell_hours": 22.0,
        "alternative_hubs": ["Antwerp-Bruges (BEANR)", "Hamburg (DEHAM)"],
    },
    "Dubai (Jebel Ali)": {
        "code": "AEJEA",
        "lat": 25.0113,
        "lon": 55.0612,
        "country": "UAE",
        "base_congestion": 0.46,
        "avg_dwell_hours": 19.5,
        "alternative_hubs": ["Khalifa Port Abu Dhabi (AEKHL)", "Salalah (OMSLL)"],
    },
    "Los Angeles": {
        "code": "USLAX",
        "lat": 33.7405,
        "lon": -118.2719,
        "country": "USA",
        "base_congestion": 0.68,
        "avg_dwell_hours": 33.5,
        "alternative_hubs": ["Long Beach (USLGB)", "Oakland (USOAK)"],
    },
}


SUPPLIERS_CATALOG: List[Dict[str, Any]] = [
    {
        "supplier_id": "SUP-101",
        "name": "Apex Semiconductor Taiwan",
        "tier": 1,
        "category": "Microelectronics & SoC",
        "country": "Taiwan",
        "lat": 24.8138,
        "lon": 120.9675,
        "on_time_rate": 0.68,
        "defect_rate_ppm": 1420,
        "financial_stability_score": 0.61,
        "geopolitical_risk_index": 0.79,
        "lead_time_days_avg": 24.5,
        "lead_time_std_days": 6.8,
        "single_source": True,
        "backup_supplier_id": "SUP-104",
        "annual_spend_usd": 14500000,
    },
    {
        "supplier_id": "SUP-102",
        "name": "Shenzhen Precision Optics Co.",
        "tier": 1,
        "category": "Sensors & LiDAR Modules",
        "country": "China",
        "lat": 22.5431,
        "lon": 114.0579,
        "on_time_rate": 0.74,
        "defect_rate_ppm": 980,
        "financial_stability_score": 0.75,
        "geopolitical_risk_index": 0.66,
        "lead_time_days_avg": 19.0,
        "lead_time_std_days": 5.1,
        "single_source": False,
        "backup_supplier_id": "SUP-105",
        "annual_spend_usd": 9800000,
    },
    {
        "supplier_id": "SUP-103",
        "name": "RheinMetall Industrial Components",
        "tier": 1,
        "category": "High-Torque Actuators",
        "country": "Germany",
        "lat": 51.2277,
        "lon": 6.7735,
        "on_time_rate": 0.94,
        "defect_rate_ppm": 210,
        "financial_stability_score": 0.92,
        "geopolitical_risk_index": 0.22,
        "lead_time_days_avg": 14.0,
        "lead_time_std_days": 1.6,
        "single_source": False,
        "backup_supplier_id": "SUP-106",
        "annual_spend_usd": 11200000,
    },
    {
        "supplier_id": "SUP-104",
        "name": "Bharat Forge & Silicon Systems",
        "tier": 1,
        "category": "Microelectronics & Power Chiplets",
        "country": "India",
        "lat": 18.5204,
        "lon": 73.8567,
        "on_time_rate": 0.91,
        "defect_rate_ppm": 340,
        "financial_stability_score": 0.89,
        "geopolitical_risk_index": 0.25,
        "lead_time_days_avg": 9.5,
        "lead_time_std_days": 1.4,
        "single_source": False,
        "backup_supplier_id": "SUP-103",
        "annual_spend_usd": 8400000,
    },
    {
        "supplier_id": "SUP-105",
        "name": "ASEAN Integrated Logistics & Parts",
        "tier": 2,
        "category": "Sensors & LiDAR Modules",
        "country": "Singapore",
        "lat": 1.3521,
        "lon": 103.8198,
        "on_time_rate": 0.89,
        "defect_rate_ppm": 410,
        "financial_stability_score": 0.87,
        "geopolitical_risk_index": 0.29,
        "lead_time_days_avg": 12.0,
        "lead_time_std_days": 2.0,
        "single_source": False,
        "backup_supplier_id": "SUP-104",
        "annual_spend_usd": 6900000,
    },
    {
        "supplier_id": "SUP-106",
        "name": "RedSea PetroChem & Polymers",
        "tier": 2,
        "category": "Battery Casings & Polymers",
        "country": "UAE",
        "lat": 25.2048,
        "lon": 55.2708,
        "on_time_rate": 0.62,
        "defect_rate_ppm": 1650,
        "financial_stability_score": 0.58,
        "geopolitical_risk_index": 0.82,
        "lead_time_days_avg": 21.0,
        "lead_time_std_days": 7.4,
        "single_source": True,
        "backup_supplier_id": "SUP-104",
        "annual_spend_usd": 7300000,
    },
]


WAREHOUSES_CATALOG: List[Dict[str, Any]] = [
    {
        "warehouse_id": "WH-MUM-01",
        "name": "Mumbai Bhiwandi Mega Fulfillment Hub",
        "city": "Mumbai",
        "lat": 19.2813,
        "lon": 73.0483,
        "sku_id": "SKU-SOC-900",
        "sku_name": "Edge AI Neural Compute Module",
        "current_stock_units": 1420,
        "daily_demand_mean": 240,
        "daily_demand_std": 55,
        "safety_stock_units": 960,
        "reorder_point_units": 1800,
        "lead_time_days": 7,
        "inbound_shipment_id": "SH-1042",
        "inbound_units": 3000,
        "inbound_eta_hours": 8.0,
        "inbound_delay_hours": 11.0,
        "unit_margin_usd": 145.0,
    },
    {
        "warehouse_id": "WH-BLR-02",
        "name": "Bengaluru Aerospace & EV Assembly DC",
        "city": "Bengaluru",
        "lat": 12.9716,
        "lon": 77.5946,
        "sku_id": "SKU-LID-420",
        "sku_name": "Solid-State Automotive LiDAR Array",
        "current_stock_units": 680,
        "daily_demand_mean": 165,
        "daily_demand_std": 42,
        "safety_stock_units": 600,
        "reorder_point_units": 1200,
        "lead_time_days": 9,
        "inbound_shipment_id": "SH-1045",
        "inbound_units": 2200,
        "inbound_eta_hours": 36.0,
        "inbound_delay_hours": 18.5,
        "unit_margin_usd": 210.0,
    },
    {
        "warehouse_id": "WH-DEL-03",
        "name": "NCR Gurugram Northern Distribution Center",
        "city": "New Delhi",
        "lat": 28.4595,
        "lon": 77.0266,
        "sku_id": "SKU-ACT-710",
        "sku_name": "Precision Servo Actuator Pack",
        "current_stock_units": 3150,
        "daily_demand_mean": 190,
        "daily_demand_std": 30,
        "safety_stock_units": 750,
        "reorder_point_units": 1400,
        "lead_time_days": 6,
        "inbound_shipment_id": "SH-1048",
        "inbound_units": 1800,
        "inbound_eta_hours": 14.0,
        "inbound_delay_hours": 1.2,
        "unit_margin_usd": 95.0,
    },
    {
        "warehouse_id": "WH-MAA-04",
        "name": "Chennai Sriperumbudur Electronics Hub",
        "city": "Chennai",
        "lat": 12.9675,
        "lon": 79.9419,
        "sku_id": "SKU-BAT-300",
        "sku_name": "Thermal-Shield Battery Enclosure",
        "current_stock_units": 890,
        "daily_demand_mean": 210,
        "daily_demand_std": 48,
        "safety_stock_units": 840,
        "reorder_point_units": 1550,
        "lead_time_days": 8,
        "inbound_shipment_id": "SH-1051",
        "inbound_units": 2500,
        "inbound_eta_hours": 22.0,
        "inbound_delay_hours": 9.4,
        "unit_margin_usd": 125.0,
    },
    {
        "warehouse_id": "WH-RTM-05",
        "name": "Rotterdam Euro-Gate Fulfillment Center",
        "city": "Rotterdam",
        "lat": 51.9244,
        "lon": 4.4777,
        "sku_id": "SKU-SOC-900",
        "sku_name": "Edge AI Neural Compute Module",
        "current_stock_units": 4200,
        "daily_demand_mean": 220,
        "daily_demand_std": 35,
        "safety_stock_units": 900,
        "reorder_point_units": 1650,
        "lead_time_days": 5,
        "inbound_shipment_id": "SH-1054",
        "inbound_units": 2800,
        "inbound_eta_hours": 48.0,
        "inbound_delay_hours": 2.0,
        "unit_margin_usd": 155.0,
    },
]


def generate_historical_shipments(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic historical shipment feature records for training
    XGBoost, LightGBM, and CatBoost delay classification and regression models.
    """
    rng = np.random.default_rng(seed)

    port_congestion = rng.beta(2.2, 2.5, size=n_samples)
    weather_severity = rng.beta(1.5, 4.0, size=n_samples)
    customs_hold_prob = rng.beta(1.4, 5.0, size=n_samples)
    carrier_reliability = rng.uniform(0.55, 0.99, size=n_samples)
    supplier_risk_score = rng.uniform(0.15, 0.90, size=n_samples)
    distance_remaining_km = rng.uniform(50.0, 6500.0, size=n_samples)
    scheduled_eta_hours = distance_remaining_km / rng.uniform(28.0, 42.0, size=n_samples)
    vessel_speed_knots = rng.normal(17.5, 3.2, size=n_samples).clip(6.0, 25.0)
    route_deviation_km = rng.exponential(28.0, size=n_samples).clip(0.0, 450.0)
    ais_signal_dropouts = rng.poisson(1.2, size=n_samples)
    cargo_value_usd = rng.uniform(80000, 1800000, size=n_samples)

    # Underlying structural equation for delay probability and hours
    logit = (
        -2.35
        + 3.45 * port_congestion
        + 2.15 * weather_severity
        + 1.95 * customs_hold_prob
        - 2.10 * (carrier_reliability - 0.75)
        + 1.15 * supplier_risk_score
        + 0.0045 * route_deviation_km
        - 0.08 * (vessel_speed_knots - 16.0)
        + rng.normal(0, 0.25, size=n_samples)
    )
    prob_delay = 1.0 / (1.0 + np.exp(-logit))
    is_delayed = (prob_delay > 0.50).astype(int)

    expected_delay_hours = np.maximum(
        0.0,
        (
            12.8 * (port_congestion ** 1.35)
            + 8.4 * weather_severity
            + 7.2 * customs_hold_prob
            + 4.5 * (1.0 - carrier_reliability)
            + 0.018 * route_deviation_km
            - 1.8
            + rng.normal(0, 0.8, size=n_samples)
        ),
    )
    # Zero out negligible delays when risk is very low
    expected_delay_hours = np.where(prob_delay < 0.22, expected_delay_hours * 0.2, expected_delay_hours)

    df = pd.DataFrame(
        {
            "port_congestion_index": np.round(port_congestion, 4),
            "weather_severity": np.round(weather_severity, 4),
            "customs_hold_prob": np.round(customs_hold_prob, 4),
            "carrier_reliability": np.round(carrier_reliability, 4),
            "supplier_risk_score": np.round(supplier_risk_score, 4),
            "distance_remaining_km": np.round(distance_remaining_km, 2),
            "scheduled_eta_hours": np.round(scheduled_eta_hours, 2),
            "vessel_speed_knots": np.round(vessel_speed_knots, 2),
            "route_deviation_km": np.round(route_deviation_km, 2),
            "ais_signal_dropouts": ais_signal_dropouts,
            "cargo_value_usd": np.round(cargo_value_usd, 2),
            "is_delayed": is_delayed,
            "delay_hours": np.round(expected_delay_hours, 2),
        }
    )
    return df


def get_live_shipments() -> List[Dict[str, Any]]:
    """
    Returns the live fleet of tracked shipments, anchored by SH-1042
    (the Mumbai port congestion disruption scenario) plus global active shipments.
    """
    return [
        {
            "shipment_id": "SH-1042",
            "origin_port": "Singapore",
            "destination_port": "Mumbai (JNPT)",
            "origin_lat": 1.2644,
            "origin_lon": 103.8222,
            "current_lat": 17.6200,
            "current_lon": 72.1800,
            "dest_lat": 18.9490,
            "dest_lon": 72.9512,
            "carrier": "Maersk Indo-Pacific Express",
            "transport_mode": "Ocean Vessel",
            "supplier_id": "SUP-101",
            "warehouse_id": "WH-MUM-01",
            "sku_id": "SKU-SOC-900",
            "cargo_description": "Edge AI Neural Compute Modules (3,000 units)",
            "cargo_value_usd": 680000.0,
            "sla_penalty_per_hour_usd": 3200.0,
            "scheduled_eta_hours": 8.0,
            "distance_remaining_km": 245.0,
            "vessel_speed_knots": 11.4,
            "port_congestion_index": 0.89,
            "weather_severity": 0.28,
            "customs_hold_prob": 0.21,
            "carrier_reliability": 0.79,
            "supplier_risk_score": 0.74,
            "route_deviation_km": 42.0,
            "ais_signal_dropouts": 1,
            "status": "CRITICAL_CONGESTION_ALERT",
            "last_updated": datetime.utcnow().isoformat() + "Z",
        },
        {
            "shipment_id": "SH-1045",
            "origin_port": "Shanghai",
            "destination_port": "Chennai",
            "origin_lat": 31.2304,
            "origin_lon": 121.4737,
            "current_lat": 9.8500,
            "current_lon": 83.4500,
            "dest_lat": 13.0827,
            "dest_lon": 80.2707,
            "carrier": "COSCO Maritime Line",
            "transport_mode": "Ocean Vessel",
            "supplier_id": "SUP-102",
            "warehouse_id": "WH-BLR-02",
            "sku_id": "SKU-LID-420",
            "cargo_description": "Solid-State Automotive LiDAR Arrays (2,200 units)",
            "cargo_value_usd": 920000.0,
            "sla_penalty_per_hour_usd": 4100.0,
            "scheduled_eta_hours": 36.0,
            "distance_remaining_km": 690.0,
            "vessel_speed_knots": 10.2,
            "port_congestion_index": 0.64,
            "weather_severity": 0.82,
            "customs_hold_prob": 0.48,
            "carrier_reliability": 0.71,
            "supplier_risk_score": 0.68,
            "route_deviation_km": 185.0,
            "ais_signal_dropouts": 5,
            "status": "WEATHER_AND_ANOMALY_RISK",
            "last_updated": datetime.utcnow().isoformat() + "Z",
        },
        {
            "shipment_id": "SH-1048",
            "origin_port": "Dubai (Jebel Ali)",
            "destination_port": "Mundra",
            "origin_lat": 25.0113,
            "origin_lon": 55.0612,
            "current_lat": 23.4100,
            "current_lon": 66.8200,
            "dest_lat": 22.7381,
            "dest_lon": 69.7041,
            "carrier": "Hapag-Lloyd Arabian Express",
            "transport_mode": "Ocean Vessel",
            "supplier_id": "SUP-103",
            "warehouse_id": "WH-DEL-03",
            "sku_id": "SKU-ACT-710",
            "cargo_description": "Precision Servo Actuator Packs (1,800 units)",
            "cargo_value_usd": 410000.0,
            "sla_penalty_per_hour_usd": 1500.0,
            "scheduled_eta_hours": 14.0,
            "distance_remaining_km": 310.0,
            "vessel_speed_knots": 19.2,
            "port_congestion_index": 0.24,
            "weather_severity": 0.12,
            "customs_hold_prob": 0.10,
            "carrier_reliability": 0.95,
            "supplier_risk_score": 0.22,
            "route_deviation_km": 8.0,
            "ais_signal_dropouts": 0,
            "status": "ON_SCHEDULE",
            "last_updated": datetime.utcnow().isoformat() + "Z",
        },
        {
            "shipment_id": "SH-1051",
            "origin_port": "Dubai (Jebel Ali)",
            "destination_port": "Chennai",
            "origin_lat": 25.0113,
            "origin_lon": 55.0612,
            "current_lat": 11.4000,
            "current_lon": 76.1000,
            "dest_lat": 13.0827,
            "dest_lon": 80.2707,
            "carrier": "MSC Global Freight",
            "transport_mode": "Ocean Vessel",
            "supplier_id": "SUP-106",
            "warehouse_id": "WH-MAA-04",
            "sku_id": "SKU-BAT-300",
            "cargo_description": "Thermal-Shield Battery Enclosures (2,500 units)",
            "cargo_value_usd": 540000.0,
            "sla_penalty_per_hour_usd": 2400.0,
            "scheduled_eta_hours": 22.0,
            "distance_remaining_km": 480.0,
            "vessel_speed_knots": 13.5,
            "port_congestion_index": 0.56,
            "weather_severity": 0.35,
            "customs_hold_prob": 0.76,
            "carrier_reliability": 0.69,
            "supplier_risk_score": 0.81,
            "route_deviation_km": 112.0,
            "ais_signal_dropouts": 4,
            "status": "CUSTOMS_HOLD_WARNING",
            "last_updated": datetime.utcnow().isoformat() + "Z",
        },
        {
            "shipment_id": "SH-1054",
            "origin_port": "Singapore",
            "destination_port": "Rotterdam",
            "origin_lat": 1.2644,
            "origin_lon": 103.8222,
            "current_lat": 48.2000,
            "current_lon": -4.5000,
            "dest_lat": 51.9244,
            "dest_lon": 4.4777,
            "carrier": "CMA CGM Euro-Asia",
            "transport_mode": "Ocean Vessel",
            "supplier_id": "SUP-105",
            "warehouse_id": "WH-RTM-05",
            "sku_id": "SKU-SOC-900",
            "cargo_description": "Edge AI Neural Compute Modules (2,800 units)",
            "cargo_value_usd": 640000.0,
            "sla_penalty_per_hour_usd": 2100.0,
            "scheduled_eta_hours": 48.0,
            "distance_remaining_km": 780.0,
            "vessel_speed_knots": 18.6,
            "port_congestion_index": 0.31,
            "weather_severity": 0.19,
            "customs_hold_prob": 0.14,
            "carrier_reliability": 0.91,
            "supplier_risk_score": 0.28,
            "route_deviation_km": 14.0,
            "ais_signal_dropouts": 0,
            "status": "ON_SCHEDULE",
            "last_updated": datetime.utcnow().isoformat() + "Z",
        },
        {
            "shipment_id": "SH-1059",
            "origin_port": "Shanghai",
            "destination_port": "Los Angeles",
            "origin_lat": 31.2304,
            "origin_lon": 121.4737,
            "current_lat": 34.1000,
            "current_lon": -123.5000,
            "dest_lat": 33.7405,
            "dest_lon": -118.2719,
            "carrier": "ONE Transpacific",
            "transport_mode": "Ocean Vessel",
            "supplier_id": "SUP-101",
            "warehouse_id": "WH-MUM-01",
            "sku_id": "SKU-SOC-900",
            "cargo_description": "High-Density Wafer Interconnects (4,000 units)",
            "cargo_value_usd": 1250000.0,
            "sla_penalty_per_hour_usd": 5200.0,
            "scheduled_eta_hours": 19.0,
            "distance_remaining_km": 460.0,
            "vessel_speed_knots": 12.1,
            "port_congestion_index": 0.77,
            "weather_severity": 0.44,
            "customs_hold_prob": 0.35,
            "carrier_reliability": 0.76,
            "supplier_risk_score": 0.74,
            "route_deviation_km": 95.0,
            "ais_signal_dropouts": 2,
            "status": "ELEVATED_DELAY_RISK",
            "last_updated": datetime.utcnow().isoformat() + "Z",
        },
    ]


def generate_demand_history(
    sku_id: str = "SKU-SOC-900", days_history: int = 120, seed: int = 42
) -> pd.DataFrame:
    """
    Generates daily historical demand time-series with trend, weekly seasonality,
    monthly promotional spikes, and exogenous macro indices for Prophet & TFT models.
    """
    rng = np.random.default_rng(seed + abs(hash(sku_id)) % 1000)
    base_map = {
        "SKU-SOC-900": 240.0,
        "SKU-LID-420": 165.0,
        "SKU-ACT-710": 190.0,
        "SKU-BAT-300": 210.0,
    }
    base_demand = base_map.get(sku_id, 200.0)

    end_date = datetime.utcnow().date()
    dates = [end_date - timedelta(days=days_history - 1 - i) for i in range(days_history)]

    records = []
    for idx, dt in enumerate(dates):
        trend = idx * 0.42
        weekly = 28.0 * math.sin(2 * math.pi * dt.weekday() / 7.0)
        monthly = 19.0 * math.cos(2 * math.pi * dt.day / 30.5)
        promo_flag = 1 if (dt.day in (1, 15, 28)) else 0
        promo_boost = 52.0 * promo_flag
        macro_pmi_index = 51.5 + 2.5 * math.sin(2 * math.pi * idx / 60.0)
        noise = rng.normal(0, 14.0)

        demand = max(40.0, base_demand + trend + weekly + monthly + promo_boost + noise)
        records.append(
            {
                "ds": pd.Timestamp(dt),
                "sku_id": sku_id,
                "y": round(demand, 1),
                "promo_flag": promo_flag,
                "macro_pmi_index": round(macro_pmi_index, 2),
                "time_idx": idx,
            }
        )
    return pd.DataFrame(records)
