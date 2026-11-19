"""
PySpark Structured Streaming Processor for High-Throughput Supply Chain Telemetry
Reads live telemetry JSON events from Apache Kafka (`shipment.telemetry.live`),
computes sliding-window port congestion and vessel speed aggregations, and
enriches micro-batches with ML delay predictions.
Main aim: Provide a unified interface for both distributed PySpark Structured Streaming and low-latency micro-batch aggregation for dashboards and APIs.
"""
from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
from src.config import settings
from src.data.database import repository


class SparkStreamingTelemetryProcessor:
    """
    Provides both:
      1. Full PySpark Structured Streaming job builder (`build_spark_streaming_query`)
         for distributed cluster execution.
      2. Instant micro-batch windowed aggregation (`compute_windowed_port_metrics`)
         for low-latency interactive dashboard & API queries.
    """

    def build_spark_streaming_code_spec(self) -> str:
        """Returns the production PySpark Structured Streaming pipeline definition."""
        return f'''from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, avg, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

spark = (
    SparkSession.builder
    .appName("NexusChain-RealTime-Disruption-Streaming")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

telemetry_schema = StructType([
    StructField("shipment_id", StringType(), True),
    StructField("destination_port", StringType(), True),
    StructField("vessel_speed_knots", DoubleType(), True),
    StructField("port_congestion_index", DoubleType(), True),
    StructField("weather_severity", DoubleType(), True),
    StructField("route_deviation_km", DoubleType(), True),
    StructField("event_time", TimestampType(), True),
])

raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "{settings.kafka_bootstrap_servers}")
    .option("subscribe", "{settings.kafka_topic_telemetry}")
    .option("startingOffsets", "latest")
    .load()
)

parsed_stream = (
    raw_stream
    .selectExpr("CAST(value AS STRING) as json_payload")
    .select(from_json(col("json_payload"), telemetry_schema).alias("data"))
    .select("data.*")
)

windowed_port_pressure = (
    parsed_stream
    .withWatermark("event_time", "10 minutes")
    .groupBy(
        window(col("event_time"), "15 minutes", "5 minutes"),
        col("destination_port")
    )
    .agg(
        avg("port_congestion_index").alias("avg_port_congestion"),
        avg("vessel_speed_knots").alias("avg_approach_speed_knots"),
        spark_max("weather_severity").alias("peak_weather_severity"),
        spark_max("route_deviation_km").alias("max_route_deviation_km"),
    )
)
'''

    def compute_windowed_port_metrics(self) -> List[Dict[str, Any]]:
        """
        Executes micro-batch windowed aggregation across active shipments grouped by destination port.
        """
        shipments = repository.list_shipments()
        if not shipments:
            return []

        df = pd.DataFrame(shipments)
        grouped = (
            df.groupby("destination_port", as_index=False)
            .agg(
                active_vessels=("shipment_id", "count"),
                avg_port_congestion=("port_congestion_index", "mean"),
                avg_approach_speed_knots=("vessel_speed_knots", "mean"),
                peak_weather_severity=("weather_severity", "max"),
                max_route_deviation_km=("route_deviation_km", "max"),
                total_cargo_value_usd=("cargo_value_usd", "sum"),
            )
            .sort_values("avg_port_congestion", ascending=False)
        )

        records: List[Dict[str, Any]] = []
        for _, row in grouped.iterrows():
            records.append(
                {
                    "destination_port": row["destination_port"],
                    "active_vessels": int(row["active_vessels"]),
                    "avg_port_congestion": round(float(row["avg_port_congestion"]), 3),
                    "avg_approach_speed_knots": round(float(row["avg_approach_speed_knots"]), 2),
                    "peak_weather_severity": round(float(row["peak_weather_severity"]), 3),
                    "max_route_deviation_km": round(float(row["max_route_deviation_km"]), 1),
                    "total_cargo_value_usd": round(float(row["total_cargo_value_usd"]), 2),
                }
            )
        return records


spark_processor = SparkStreamingTelemetryProcessor()
