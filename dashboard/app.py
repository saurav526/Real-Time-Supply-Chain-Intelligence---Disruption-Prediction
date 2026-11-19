"""
NexusChain AI — Executive Streamlit + Plotly Command Center
Real-Time Supply Chain Intelligence & Disruption Prediction Platform
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.database import repository
from src.data.generator import PORTS_AND_HUBS
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

st.set_page_config(
    page_title="NexusChain AI | Real-Time Supply Chain Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .alert-box-critical {
        background: linear-gradient(90deg, rgba(220,38,38,0.18) 0%, rgba(30,41,59,0.92) 100%);
        border-left: 6px solid #ef4444;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .alert-box-title {
        color: #fca5a5;
        font-weight: 700;
        font-size: 0.92rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .alert-box-body {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 500;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar() -> str:
    with st.sidebar:
        st.title("🚢 NexusChain AI")
        st.caption("Real-Time Supply Chain Intelligence & Disruption Prediction")
        st.divider()

        shipments = repository.list_shipments()
        shipment_ids = [s["shipment_id"] for s in shipments]
        selected_shipment = st.selectbox(
            "Active Focus Shipment",
            options=shipment_ids,
            index=0,
            help="Select a live shipment to inspect across prediction, rerouting, and financial tabs.",
        )

        st.subheader("System Telemetry Status")
        st.markdown(f"- **DB Engine:** `{repository.db_backend_name}`")
        st.markdown(f"- **Cache Layer:** `{repository.cache.backend_name}`")
        st.markdown(f"- **Event Bus:** `{kafka_pipeline.broker_mode}`")
        st.markdown("- **ML Ensemble:** `XGBoost + LightGBM + CatBoost`")
        st.markdown("- **Forecasters:** `TFT + Prophet`")

        if st.button("🔄 Reset Telemetry to Baseline", use_container_width=True):
            repository.reset_shipments_to_default()
            st.toast("All shipments reset to baseline telemetry!", icon="✅")
            st.rerun()

        return selected_shipment


def main() -> None:
    selected_shipment_id = render_sidebar()

    st.title("🌐 Real-Time Supply Chain Intelligence & Disruption Prediction")
    st.caption(
        "Continuous Multi-Tier Supply Chain Monitoring • Ensemble ETA & Delay Prediction • "
        "TFT/Prophet Demand Forecasting • Warehouse Stockout Prevention • Prescriptive Rerouting"
    )

    shipments = repository.list_shipments()
    enriched_shipments = []
    for s in shipments:
        pred = delay_predictor.predict_shipment(s)
        enriched_shipments.append({**s, "prediction": pred})

    focus_ship = next(
        (s for s in enriched_shipments if s["shipment_id"] == selected_shipment_id),
        enriched_shipments[0],
    )
    focus_pred = focus_ship["prediction"]

    # Real-Time Live Alert Banner
    st.markdown(
        f"""
        <div class="alert-box-critical">
            <div class="alert-box-title">⚡ Live Disruption Intelligence Signal — {focus_ship['shipment_id']} ({focus_ship['origin_port']} → {focus_ship['destination_port']})</div>
            <div class="alert-box-body">"{focus_pred['alert_message']}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Executive KPI Row
    portfolio_fin = recommendation_engine.compute_portfolio_financial_summary()
    warehouses = stockout_predictor.evaluate_warehouses()
    suppliers = supplier_risk_engine.analyze_suppliers()
    anomalies = anomaly_detector.detect_anomalies()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        f"{focus_ship['shipment_id']} Delay Prob.",
        f"{focus_pred['delay_probability_pct']}%",
        f"+{focus_pred['expected_delay_hours']}h Expected Delay",
        delta_color="inverse",
    )
    k2.metric(
        "Critical / High-Risk Shipments",
        f"{sum(1 for s in enriched_shipments if s['prediction']['delay_probability'] >= 0.50)} / {len(enriched_shipments)}",
        f"{sum(1 for a in anomalies if a['is_anomaly'])} Pattern Anomalies",
        delta_color="inverse",
    )
    k3.metric(
        "Warehouse Stockout Alerts (7d)",
        f"{sum(1 for w in warehouses if w['stockout_probability_7d'] >= 0.50)} Hubs",
        f"Peak: {warehouses[0]['warehouse_id']} ({warehouses[0]['stockout_probability_7d_pct']}%)",
        delta_color="inverse",
    )
    k4.metric(
        "High-Risk Tier-1/2 Suppliers",
        f"{sum(1 for s in suppliers if s['risk_level'] == 'HIGH_RISK')} Suppliers",
        f"Top: {suppliers[0]['supplier_id']} ({suppliers[0]['composite_risk_pct']}%)",
        delta_color="inverse",
    )
    k5.metric(
        "Prescriptive Net Savings",
        f"${portfolio_fin['total_prescriptive_net_savings_usd']:,.0f}",
        f"vs ${portfolio_fin['total_unmitigated_disruption_exposure_usd']:,.0f} Exposure",
    )

    tabs = st.tabs(
        [
            "🗺️ Global Control Tower",
            "🎯 Delay Predictor & Simulator",
            "📈 Demand Forecasting (TFT + Prophet)",
            "🏭 Warehouse Stockout Risk",
            "🛡️ Supplier Risk & Anomalies",
            "🧭 Rerouting & Financial Impact",
            "⚙️ Streaming & MLOps (MLflow/Evidently)",
        ]
    )

    # =========================================================================
    # TAB 1: GLOBAL CONTROL TOWER
    # =========================================================================
    with tabs[0]:
        col_map, col_details = st.columns([1.65, 1.0])

        with col_map:
            st.subheader("Live Maritime & Intermodal Corridor Telemetry")
            fig_map = go.Figure()

            # Plot shipping corridors and current vessel positions
            for s in enriched_shipments:
                p = s["prediction"]
                color = (
                    "#ef4444"
                    if p["delay_probability"] >= 0.65
                    else ("#f59e0b" if p["delay_probability"] >= 0.40 else "#10b981")
                )
                # Corridor line
                fig_map.add_trace(
                    go.Scattergeo(
                        lon=[s["origin_lon"], s["current_lon"], s["dest_lon"]],
                        lat=[s["origin_lat"], s["current_lat"], s["dest_lat"]],
                        mode="lines",
                        line=dict(width=2.2, color=color),
                        name=f"{s['shipment_id']} Route",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                # Live vessel marker
                fig_map.add_trace(
                    go.Scattergeo(
                        lon=[s["current_lon"]],
                        lat=[s["current_lat"]],
                        mode="markers+text",
                        text=[s["shipment_id"]],
                        textposition="top center",
                        marker=dict(
                            size=13 if s["shipment_id"] == selected_shipment_id else 10,
                            color=color,
                            line=dict(width=2, color="#ffffff"),
                            symbol="diamond",
                        ),
                        name=f"{s['shipment_id']} ({p['delay_probability_pct']}% risk)",
                        hovertemplate=(
                            f"<b>{s['shipment_id']}</b> ({s['carrier']})<br>"
                            f"Route: {s['origin_port']} → {s['destination_port']}<br>"
                            f"Delay Prob: {p['delay_probability_pct']}% | Expected Delay: {p['expected_delay_hours']}h<br>"
                            f"Primary Cause: {p['primary_cause']}<extra></extra>"
                        ),
                    )
                )

            # Plot ports with congestion halos
            port_lats, port_lons, port_names, port_sizes = [], [], [], []
            for p_name, meta in PORTS_AND_HUBS.items():
                port_lats.append(meta["lat"])
                port_lons.append(meta["lon"])
                port_names.append(f"{p_name} (Congestion: {int(meta['base_congestion']*100)}%)")
                port_sizes.append(10 + meta["base_congestion"] * 12)

            fig_map.add_trace(
                go.Scattergeo(
                    lon=port_lons,
                    lat=port_lats,
                    mode="markers",
                    text=port_names,
                    marker=dict(
                        size=port_sizes,
                        color="rgba(56, 189, 248, 0.35)",
                        line=dict(width=1.5, color="#38bdf8"),
                    ),
                    name="Logistics Hubs & Ports",
                )
            )

            fig_map.update_geos(
                projection_type="natural earth",
                showland=True,
                landcolor="#1e293b",
                showocean=True,
                oceancolor="#0f172a",
                showcountries=True,
                countrycolor="#334155",
                lataxis_range=[-5, 60],
                lonaxis_range=[-130, 135],
            )
            fig_map.update_layout(
                height=470,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.05),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with col_details:
            st.subheader(f"Focus Telemetry: {focus_ship['shipment_id']}")
            st.info(focus_pred["alert_message"])

            m_c1, m_c2 = st.columns(2)
            m_c1.metric("Scheduled ETA", f"{focus_ship['scheduled_eta_hours']} hrs")
            m_c2.metric(
                "Predicted Total ETA",
                f"{focus_pred['predicted_total_eta_hours']} hrs",
                f"+{focus_pred['expected_delay_hours']} hrs",
                delta_color="inverse",
            )

            drv_df = pd.DataFrame(
                [
                    {"Signal Driver": k.title(), "Impact Score": v}
                    for k, v in focus_pred["driver_attributions"].items()
                ]
            ).sort_values("Impact Score", ascending=True)

            fig_drv = px.bar(
                drv_df,
                x="Impact Score",
                y="Signal Driver",
                orientation="h",
                title="Root-Cause Attribution Signals (SHAP Local Drivers)",
                color="Impact Score",
                color_continuous_scale="Reds",
            )
            fig_drv.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_drv, use_container_width=True)

        st.subheader("Active Global Shipment Fleet & Real-Time Predictions")
        fleet_rows = []
        for s in enriched_shipments:
            p = s["prediction"]
            fleet_rows.append(
                {
                    "Shipment ID": s["shipment_id"],
                    "Origin": s["origin_port"],
                    "Destination": s["destination_port"],
                    "Carrier": s["carrier"],
                    "Scheduled ETA (h)": s["scheduled_eta_hours"],
                    "Delay Prob (%)": f"{p['delay_probability_pct']}%",
                    "Expected Delay (h)": p["expected_delay_hours"],
                    "Primary Disruption Cause": p["primary_cause"],
                    "Severity": p["severity"],
                    "Recommended Action": p["recommended_action"],
                }
            )
        st.dataframe(pd.DataFrame(fleet_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: DELAY PREDICTOR & REAL-TIME SIMULATOR
    # =========================================================================
    with tabs[1]:
        st.subheader(
            f"Interactive Telemetry Simulator & Multi-Model Ensemble ({focus_ship['shipment_id']})"
        )
        sim_col, res_col = st.columns([1.05, 1.35])

        with sim_col:
            st.markdown("##### Inject Live Tracking Telemetry Update (Kafka Event Stream)")
            with st.form("telemetry_sim_form"):
                new_cong = st.slider(
                    "Destination Port Congestion Index",
                    0.0,
                    1.0,
                    float(focus_ship["port_congestion_index"]),
                    0.01,
                )
                new_weather = st.slider(
                    "Maritime Weather Severity Index",
                    0.0,
                    1.0,
                    float(focus_ship["weather_severity"]),
                    0.01,
                )
                new_customs = st.slider(
                    "Customs Inspection Hold Probability",
                    0.0,
                    1.0,
                    float(focus_ship["customs_hold_prob"]),
                    0.01,
                )
                new_speed = st.slider(
                    "Live Vessel Speed (Knots)",
                    5.0,
                    25.0,
                    float(focus_ship["vessel_speed_knots"]),
                    0.2,
                )
                new_dev = st.slider(
                    "Route Deviation from Corridor (km)",
                    0.0,
                    400.0,
                    float(focus_ship["route_deviation_km"]),
                    5.0,
                )
                new_eta = st.number_input(
                    "Scheduled Remaining ETA (Hours)",
                    min_value=1.0,
                    max_value=120.0,
                    value=float(focus_ship["scheduled_eta_hours"]),
                    step=1.0,
                )
                submitted = st.form_submit_button(
                    "📡 Publish Telemetry Event to Kafka & Re-Predict",
                    use_container_width=True,
                )
                if submitted:
                    kafka_pipeline.publish_telemetry_update(
                        focus_ship["shipment_id"],
                        {
                            "port_congestion_index": new_cong,
                            "weather_severity": new_weather,
                            "customs_hold_prob": new_customs,
                            "vessel_speed_knots": new_speed,
                            "route_deviation_km": new_dev,
                            "scheduled_eta_hours": new_eta,
                        },
                    )
                    st.success(
                        f"Published telemetry event to `{kafka_pipeline.telemetry_topic}` and updated predictions!"
                    )
                    st.rerun()

        with res_col:
            st.markdown("##### Ensemble Model Breakdown (XGBoost vs LightGBM vs CatBoost)")
            mb = focus_pred["model_breakdown"]
            mb_df = pd.DataFrame(
                [
                    {
                        "Model": m_name,
                        "Delay Probability (%)": round(vals["delay_probability"] * 100, 1),
                        "Expected Delay (Hours)": vals["expected_delay_hours"],
                    }
                    for m_name, vals in mb.items()
                ]
                + [
                    {
                        "Model": "Ensemble (Blended)",
                        "Delay Probability (%)": round(focus_pred["delay_probability"] * 100, 1),
                        "Expected Delay (Hours)": focus_pred["expected_delay_hours"],
                    }
                ]
            )
            st.dataframe(mb_df, use_container_width=True, hide_index=True)

            fig_mb = go.Figure()
            fig_mb.add_trace(
                go.Bar(
                    x=mb_df["Model"],
                    y=mb_df["Delay Probability (%)"],
                    name="Delay Probability (%)",
                    marker_color=["#38bdf8", "#818cf8", "#f472b6", "#ef4444"],
                    text=mb_df["Delay Probability (%)"].apply(lambda v: f"{v}%"),
                    textposition="outside",
                )
            )
            fig_mb.update_layout(
                title="Model-by-Model Late Arrival Probability (%)",
                yaxis_range=[0, 108],
                height=310,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_mb, use_container_width=True)

            st.markdown("##### Generated Real-Time Automated Response")
            st.code(focus_pred["alert_message"], language="text")

    # =========================================================================
    # TAB 3: DEMAND FORECASTING STUDIO (TFT + PROPHET)
    # =========================================================================
    with tabs[2]:
        st.subheader("Multi-Horizon Product Demand Forecasting (Temporal Fusion Transformer & Prophet)")
        fc_c1, fc_c2 = st.columns([1, 2])
        with fc_c1:
            selected_sku = st.selectbox(
                "Select Product SKU",
                options=[
                    "SKU-SOC-900",
                    "SKU-LID-420",
                    "SKU-ACT-710",
                    "SKU-BAT-300",
                ],
                format_func=lambda x: {
                    "SKU-SOC-900": "SKU-SOC-900 — Edge AI Neural Compute Module",
                    "SKU-LID-420": "SKU-LID-420 — Solid-State Automotive LiDAR Array",
                    "SKU-ACT-710": "SKU-ACT-710 — Precision Servo Actuator Pack",
                    "SKU-BAT-300": "SKU-BAT-300 — Thermal-Shield Battery Enclosure",
                }[x],
            )
        with fc_c2:
            horizon_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=30, value=14)

        fc_data = demand_forecaster.forecast_sku(sku_id=selected_sku, horizon_days=horizon_days)

        fk1, fk2, fk3, fk4 = st.columns(4)
        fk1.metric("TFT P50 Cumulative Demand", f"{fc_data['total_predicted_demand_p50']:,.0f} units")
        fk2.metric("TFT P90 Upper-Bound Demand", f"{fc_data['total_predicted_demand_p90']:,.0f} units")
        fk3.metric("TFT Holdout MAPE", f"{fc_data['metrics']['tft_mape_pct']}%")
        fk4.metric("Prophet Holdout MAPE", f"{fc_data['metrics']['prophet_mape_pct']}%")

        hist_df = pd.DataFrame(fc_data["historical_series"])
        fcast_df = pd.DataFrame(fc_data["forecast_series"])

        fig_fc = go.Figure()
        fig_fc.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["actual_demand"],
                mode="lines+markers",
                name="Historical Actual Demand",
                line=dict(color="#94a3b8", width=2),
            )
        )
        # P10-P90 uncertainty band
        fig_fc.add_trace(
            go.Scatter(
                x=list(fcast_df["date"]) + list(fcast_df["date"])[::-1],
                y=list(fcast_df["tft_p90"]) + list(fcast_df["tft_p10"])[::-1],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.22)",
                line=dict(color="rgba(255,255,255,0)"),
                name="TFT P10–P90 Uncertainty Fan",
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=fcast_df["date"],
                y=fcast_df["tft_p50"],
                mode="lines+markers",
                name="TFT Median Forecast (P50)",
                line=dict(color="#38bdf8", width=3),
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=fcast_df["date"],
                y=fcast_df["prophet_forecast"],
                mode="lines",
                name="Prophet Baseline Forecast",
                line=dict(color="#f59e0b", width=2, dash="dash"),
            )
        )
        fig_fc.update_layout(
            title=f"{selected_sku} — {horizon_days}-Day Demand Forecast with Quantile Intervals",
            xaxis_title="Date",
            yaxis_title="Daily Demand (Units)",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            var_df = pd.DataFrame(
                [
                    {"Covariate": k, "Importance Weight": v}
                    for k, v in fc_data["variable_selection_weights"].items()
                ]
            ).sort_values("Importance Weight", ascending=True)
            fig_vw = px.bar(
                var_df,
                x="Importance Weight",
                y="Covariate",
                orientation="h",
                title="TFT Variable Selection Network Weights",
                color="Importance Weight",
                color_continuous_scale="Blues",
            )
            fig_vw.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_vw, use_container_width=True)

        with sub_c2:
            fig_decomp = go.Figure()
            fig_decomp.add_trace(
                go.Scatter(
                    x=fcast_df["date"],
                    y=fcast_df["weekly_seasonality"],
                    mode="lines+markers",
                    name="Weekly Seasonality",
                )
            )
            fig_decomp.add_trace(
                go.Bar(
                    x=fcast_df["date"],
                    y=fcast_df["promo_effect"],
                    name="Promotional Spike Effect",
                )
            )
            fig_decomp.update_layout(
                title="Prophet Seasonality & Promotional Effect Decomposition",
                height=300,
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

    # =========================================================================
    # TAB 4: WAREHOUSE STOCKOUT & INVENTORY RISK
    # =========================================================================
    with tabs[3]:
        st.subheader("Warehouse Stockout Probability & Inbound Delay Impact")
        wh_df = pd.DataFrame(
            [
                {
                    "Warehouse ID": w["warehouse_id"],
                    "Fulfillment Hub": w["name"],
                    "SKU": w["sku_id"],
                    "Current Stock": w["current_stock_units"],
                    "Daily Demand": w["daily_demand_mean"],
                    "Days of Supply": w["days_of_supply"],
                    "Inbound Shipment": w["inbound_shipment_id"],
                    "Inbound Delay (h)": w["inbound_expected_delay_hours"],
                    "7d Stockout Prob": f"{w['stockout_probability_7d_pct']}%",
                    "Risk Tier": w["risk_tier"],
                    "Financial Exposure ($)": f"${w['financial_exposure_usd']:,.0f}",
                    "Recommended Action": w["recommended_inventory_action"],
                }
                for w in warehouses
            ]
        )
        st.dataframe(wh_df, use_container_width=True, hide_index=True)

        wc1, wc2 = st.columns([1.0, 1.3])
        with wc1:
            fig_wh_bar = px.bar(
                warehouses,
                x="warehouse_id",
                y="stockout_probability_7d_pct",
                color="risk_tier",
                color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "LOW": "#10b981"},
                title="7-Day Stockout Probability by Fulfillment Hub (%)",
                labels={"warehouse_id": "Warehouse", "stockout_probability_7d_pct": "Stockout Prob (%)"},
                text="stockout_probability_7d_pct",
            )
            fig_wh_bar.update_layout(height=360, yaxis_range=[0, 105])
            st.plotly_chart(fig_wh_bar, use_container_width=True)

        with wc2:
            sel_wh_id = st.selectbox(
                "Inspect 10-Day Inventory Depletion Curve",
                options=[w["warehouse_id"] for w in warehouses],
                format_func=lambda wid: f"{wid} — {next(w['name'] for w in warehouses if w['warehouse_id'] == wid)}",
            )
            sel_wh = next(w for w in warehouses if w["warehouse_id"] == sel_wh_id)
            traj_df = pd.DataFrame(sel_wh["inventory_trajectory"])

            fig_traj = go.Figure()
            fig_traj.add_trace(
                go.Scatter(
                    x=traj_df["day"],
                    y=traj_df["inventory_on_time"],
                    mode="lines+markers",
                    name="Nominal Inventory (On-Time Replenishment)",
                    line=dict(color="#10b981", width=3),
                )
            )
            fig_traj.add_trace(
                go.Scatter(
                    x=traj_df["day"],
                    y=traj_df["inventory_with_delay"],
                    mode="lines+markers",
                    name=f"Delayed Replenishment ({sel_wh['inbound_shipment_id']} +{sel_wh['inbound_expected_delay_hours']}h)",
                    line=dict(color="#ef4444", width=3),
                )
            )
            fig_traj.add_trace(
                go.Scatter(
                    x=traj_df["day"],
                    y=traj_df["safety_stock"],
                    mode="lines",
                    name="Safety Stock Threshold",
                    line=dict(color="#f59e0b", dash="dot", width=2),
                )
            )
            fig_traj.update_layout(
                title=f"10-Day Depletion Trajectory — {sel_wh['warehouse_id']} ({sel_wh['sku_id']})",
                yaxis_title="Available Units",
                height=320,
            )
            st.plotly_chart(fig_traj, use_container_width=True)

    # =========================================================================
    # TAB 5: SUPPLIER RISK & SHIPPING ANOMALIES
    # =========================================================================
    with tabs[4]:
        sc1, sc2 = st.columns([1.25, 1.0])
        with sc1:
            st.subheader("Multi-Tier Supplier Failure & Risk Scoring")
            sup_rows = [
                {
                    "Supplier ID": s["supplier_id"],
                    "Supplier Name": s["name"],
                    "Country": s["country"],
                    "On-Time (%)": f"{int(s['on_time_rate']*100)}%",
                    "Defects (PPM)": s["defect_rate_ppm"],
                    "Risk Score": f"{s['composite_risk_pct']}%",
                    "Risk Level": s["risk_level"],
                    "Backup Supplier": f"{s['backup_supplier_name']} ({s['backup_supplier_country']})",
                    "Annual VaR ($)": f"${s['value_at_risk_usd']:,.0f}",
                }
                for s in suppliers
            ]
            st.dataframe(pd.DataFrame(sup_rows), use_container_width=True, hide_index=True)

        with sc2:
            sel_sup_id = st.selectbox(
                "Supplier 5-Axis Risk Radar Profile",
                options=[s["supplier_id"] for s in suppliers],
                format_func=lambda sid: f"{sid} — {next(s['name'] for s in suppliers if s['supplier_id'] == sid)}",
            )
            sel_sup = next(s for s in suppliers if s["supplier_id"] == sel_sup_id)
            dims = sel_sup["risk_dimensions"]
            fig_radar = go.Figure(
                data=go.Scatterpolar(
                    r=list(dims.values()) + [list(dims.values())[0]],
                    theta=list(dims.keys()) + [list(dims.keys())[0]],
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.28)",
                    line=dict(color="#ef4444", width=2),
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"{sel_sup['name']} Risk Dimensions",
                height=310,
                margin=dict(l=30, r=30, t=40, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption(f"**Mitigation:** {sel_sup['mitigation_recommendation']}")

        st.divider()
        st.subheader("Unusual Shipping Pattern & Telemetry Anomaly Detection (Isolation Forest)")
        anom_rows = [
            {
                "Shipment ID": a["shipment_id"],
                "Corridor": f"{a['origin_port']} → {a['destination_port']}",
                "Carrier": a["carrier"],
                "Anomaly Score": a["anomaly_score"],
                "Severity": a["anomaly_severity"],
                "Speed (kt)": a["vessel_speed_knots"],
                "Route Dev (km)": a["route_deviation_km"],
                "AIS Dropouts": a["ais_signal_dropouts"],
                "Detected Unusual Patterns": " • ".join(a["detected_patterns"]),
            }
            for a in anomalies
        ]
        st.dataframe(pd.DataFrame(anom_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 6: PRESCRIPTIVE REROUTING & FINANCIAL IMPACT
    # =========================================================================
    with tabs[5]:
        eval_res = recommendation_engine.evaluate_shipment_disruption(selected_shipment_id)
        fin = eval_res["financial_impact"]
        best_opt = eval_res["recommended_route_option"]

        st.subheader(
            f"Prescriptive Mitigation & Financial Impact Analysis — {selected_shipment_id} ({eval_res['destination_port']})"
        )
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Cargo Value at Risk", f"${fin['cargo_value_usd']:,.0f}")
        f2.metric(
            "Unmitigated Disruption Cost",
            f"${fin['total_unmitigated_disruption_cost_usd']:,.0f}",
            "SLA + Stockout + Demurrage",
            delta_color="inverse",
        )
        f3.metric("Recommended Reroute Cost", f"${fin['recommended_reroute_cost_usd']:,.0f}")
        f4.metric(
            "Estimated Net Savings",
            f"${fin['estimated_net_savings_usd']:,.0f}",
            f"ROI: +{best_opt['roi_pct']}%",
        )

        rc1, rc2 = st.columns([1.1, 1.2])
        with rc1:
            fig_fin = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["relative", "relative", "relative", "relative", "total", "relative", "total"],
                    x=[
                        "SLA Penalty",
                        "Stockout/Line-Stop",
                        "Demurrage",
                        "Holding Cost",
                        "Unmitigated Loss",
                        "Reroute Mitigation Saving",
                        "Residual Cost",
                    ],
                    y=[
                        fin["sla_penalty_cost_usd"],
                        fin["stockout_and_line_stop_cost_usd"],
                        fin["demurrage_and_detention_usd"],
                        fin["capital_holding_cost_usd"],
                        0,
                        -fin["estimated_net_savings_usd"],
                        0,
                    ],
                )
            )
            fig_fin.update_layout(
                title=f"Disruption Cost Breakdown & Prescriptive Savings ({selected_shipment_id})",
                height=360,
            )
            st.plotly_chart(fig_fin, use_container_width=True)

        with rc2:
            st.markdown("##### Recommended Alternative Logistics Hubs & Routes")
            st.success(
                f"**Optimal Route ({best_opt['option_id']}):** {best_opt['hub_name']} "
                f"({best_opt['mode']}) — Saves **{best_opt['transit_time_saved_hours']} hrs**, "
                f"Net Benefit: **${best_opt['net_financial_benefit_usd']:,.0f}**"
            )
            alt_df = pd.DataFrame(
                [
                    {
                        "Option ID": o["option_id"],
                        "Alternative Logistics Hub & Corridor": o["hub_name"],
                        "Transport Mode": o["mode"],
                        "Time Saved (h)": o["transit_time_saved_hours"],
                        "Reroute Cost ($)": f"${o['reroute_cost_usd']:,.0f}",
                        "Avoided Loss ($)": f"${o['avoided_disruption_loss_usd']:,.0f}",
                        "Net Benefit ($)": f"${o['net_financial_benefit_usd']:,.0f}",
                        "ROI (%)": f"{o['roi_pct']}%",
                    }
                    for o in eval_res["alternative_route_options"]
                ]
            )
            st.dataframe(alt_df, use_container_width=True, hide_index=True)

            sup_cont = eval_res["supplier_contingency"]
            st.info(
                f"**Supplier Contingency:** Current Supplier `{sup_cont['current_supplier_name']}` "
                f"({sup_cont['current_supplier_country']}) → Pre-qualified Backup: "
                f"**{sup_cont['backup_supplier_name']}** ({sup_cont['backup_supplier_country']}, "
                f"avg lead time {sup_cont['backup_lead_time_days']} days)."
            )

    # =========================================================================
    # TAB 7: STREAMING (KAFKA/PYSPARK) & MLOPS (MLFLOW/EVIDENTLY)
    # =========================================================================
    with tabs[6]:
        st.subheader("MLflow Model Registry Leaderboard & Global Feature Importance")
        reg = mlflow_registry.get_registry_overview()
        drift = drift_monitor.evaluate_drift()

        mc1, mc2 = st.columns([1.3, 1.0])
        with mc1:
            st.dataframe(
                pd.DataFrame(reg["registered_models"]),
                use_container_width=True,
                hide_index=True,
            )
        with mc2:
            imp_df = pd.DataFrame(
                [
                    {"Feature": k, "Global Importance": v}
                    for k, v in reg["global_feature_importances"].items()
                ]
            ).sort_values("Global Importance", ascending=True)
            fig_imp = px.bar(
                imp_df,
                x="Global Importance",
                y="Feature",
                orientation="h",
                title="Global Feature Importance (XGBoost / Ensemble)",
                color="Global Importance",
                color_continuous_scale="Tealgrn",
            )
            fig_imp.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()
        st.subheader("Evidently AI Telemetry Data Drift & Covariate Shift Monitor (PSI & KS-Test)")
        dk1, dk2, dk3 = st.columns(3)
        dk1.metric("Monitor Engine", "Evidently KS + PSI")
        dk2.metric("Drifted Features", f"{drift['drifted_features_count']} / 10")
        dk3.metric("Pipeline Drift Status", drift["drift_summary_status"])

        st.dataframe(
            pd.DataFrame(drift["feature_drift_reports"]),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("PySpark Structured Streaming Windowed Port Congestion Aggregations")
        win_metrics = spark_processor.compute_windowed_port_metrics()
        st.dataframe(pd.DataFrame(win_metrics), use_container_width=True, hide_index=True)

        with st.expander("View Production PySpark Structured Streaming & Kafka Consumer Pipeline Code"):
            st.code(spark_processor.build_spark_streaming_code_spec(), language="python")


if __name__ == "__main__":
    main()
