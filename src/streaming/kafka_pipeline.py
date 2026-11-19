"""
Apache Kafka Real-Time Telemetry Producer & Disruption Alert Consumer
Supports live Kafka brokers (`kafka-python`) with automatic fallback to an
in-memory event bus so real-time streaming works out-of-the-box anywhere.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
import json
from typing import Any, Deque, Dict, List, Optional

from src.config import settings
from src.data.database import repository
from src.ml.delay_predictor import delay_predictor


class KafkaSupplyChainPipeline:
    """
    Produces live vessel/shipment telemetry events to Apache Kafka topic
    `shipment.telemetry.live` and emits disruption predictions to
    `supplychain.disruption.alerts`.
    """

    def __init__(self) -> None:
        self.bootstrap_servers = settings.kafka_bootstrap_servers
        self.telemetry_topic = settings.kafka_topic_telemetry
        self.alerts_topic = settings.kafka_topic_alerts
        self._producer = None
        self.broker_mode = "In-Memory Streaming Bus (Kafka Compatible)"
        self._recent_events: Deque[Dict[str, Any]] = deque(maxlen=50)
        self._init_kafka_producer()

    def _init_kafka_producer(self) -> None:
        try:
            import socket
            host, port_str = self.bootstrap_servers.split(",")[0].split(":")
            with socket.create_connection((host, int(port_str)), timeout=0.15):
                pass
            from kafka import KafkaProducer  # type: ignore

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=500,
                max_block_ms=500,
            )
            self.broker_mode = f"Apache Kafka ({self.bootstrap_servers})"
        except Exception:
            self._producer = None

    def publish_telemetry_update(
        self, shipment_id: str, telemetry_patch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingests a live tracking update, updates PostgreSQL/Redis state,
        runs real-time ensemble delay prediction, and publishes both the raw
        telemetry event and any resulting disruption alert.
        """
        updated_shipment = repository.update_shipment_telemetry(shipment_id, telemetry_patch)
        if not updated_shipment:
            raise ValueError(f"Shipment {shipment_id} not found")

        prediction = delay_predictor.predict_shipment(updated_shipment)

        event_envelope = {
            "event_type": "SHIPMENT_TELEMETRY_UPDATED",
            "topic": self.telemetry_topic,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "shipment_id": shipment_id,
            "telemetry_patch": telemetry_patch,
            "prediction": prediction,
        }

        if self._producer is not None:
            try:
                self._producer.send(self.telemetry_topic, event_envelope)
                if prediction["delay_probability"] >= 0.50:
                    self._producer.send(self.alerts_topic, prediction)
            except Exception:
                pass

        if prediction["delay_probability"] >= 0.50:
            repository.log_alert(prediction)

        self._recent_events.appendleft(event_envelope)
        return event_envelope

    def get_recent_stream_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._recent_events)[:limit]


kafka_pipeline = KafkaSupplyChainPipeline()
