"""
Unified Data Persistence (PostgreSQL / SQLAlchemy + SQLite fallback)
and Low-Latency State Cache (Redis + In-Memory TTL fallback).
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import settings
from src.data.generator import (
    SUPPLIERS_CATALOG,
    WAREHOUSES_CATALOG,
    get_live_shipments,
)

Base = declarative_base()


class ShipmentRecord(Base):
    __tablename__ = "shipments"

    shipment_id = Column(String(32), primary_key=True)
    origin_port = Column(String(64))
    destination_port = Column(String(64))
    origin_lat = Column(Float)
    origin_lon = Column(Float)
    current_lat = Column(Float)
    current_lon = Column(Float)
    dest_lat = Column(Float)
    dest_lon = Column(Float)
    carrier = Column(String(128))
    transport_mode = Column(String(64))
    supplier_id = Column(String(32))
    warehouse_id = Column(String(32))
    sku_id = Column(String(32))
    cargo_description = Column(String(256))
    cargo_value_usd = Column(Float)
    sla_penalty_per_hour_usd = Column(Float)
    scheduled_eta_hours = Column(Float)
    distance_remaining_km = Column(Float)
    vessel_speed_knots = Column(Float)
    port_congestion_index = Column(Float)
    weather_severity = Column(Float)
    customs_hold_prob = Column(Float)
    carrier_reliability = Column(Float)
    supplier_risk_score = Column(Float)
    route_deviation_km = Column(Float)
    ais_signal_dropouts = Column(Integer)
    status = Column(String(64))
    last_updated = Column(String(64))


class DisruptionAlertRecord(Base):
    __tablename__ = "disruption_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    shipment_id = Column(String(32), index=True)
    severity = Column(String(32))
    delay_probability = Column(Float)
    expected_delay_hours = Column(Float)
    primary_cause = Column(String(128))
    recommended_action = Column(Text)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class RedisOrMemoryCache:
    """
    Wraps Redis for sub-millisecond telemetry & prediction caching,
    falling back transparently to a thread-safe in-memory TTL store.
    """

    def __init__(self, redis_url: str) -> None:
        self._memory: Dict[str, tuple[Any, float]] = {}
        self._redis = None
        self.backend_name = "In-Memory TTL Cache (Redis Fallback)"
        try:
            import redis  # type: ignore

            client = redis.from_url(redis_url, socket_connect_timeout=0.5, decode_responses=True)
            client.ping()
            self._redis = client
            self.backend_name = "Redis 7.x Cluster"
        except Exception:
            self._redis = None

    def set_json(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        payload = json.dumps(value, default=str)
        if self._redis is not None:
            try:
                self._redis.setex(key, ttl_seconds, payload)
                return
            except Exception:
                pass
        self._memory[key] = (payload, time.time() + ttl_seconds)

    def get_json(self, key: str) -> Optional[Any]:
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return json.loads(raw) if raw else None
            except Exception:
                pass
        entry = self._memory.get(key)
        if not entry:
            return None
        payload, expires_at = entry
        if time.time() > expires_at:
            self._memory.pop(key, None)
            return None
        return json.loads(payload)


class DataRepository:
    """
    Manages PostgreSQL (or SQLite local fallback) persistence and Redis caching.
    """

    def __init__(self) -> None:
        self.db_url = settings.database_url
        self.engine = self._init_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.cache = RedisOrMemoryCache(settings.redis_url)
        self._seed_initial_data()

    def _init_engine(self, url: str):
        try:
            if url.startswith("postgresql"):
                eng = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
                with eng.connect() as _:
                    pass
                self.db_backend_name = "PostgreSQL 15"
                return eng
        except Exception:
            pass
        fallback_url = f"sqlite:///{settings.database_url.split('///')[-1]}" if "sqlite" in url else "sqlite:///./data_store/supply_chain.db"
        self.db_backend_name = "SQLite (PostgreSQL Compatible Schema)"
        return create_engine(fallback_url, connect_args={"check_same_thread": False})

    def _seed_initial_data(self) -> None:
        with self.SessionLocal() as session:
            count = session.query(ShipmentRecord).count()
            if count == 0:
                for item in get_live_shipments():
                    session.add(ShipmentRecord(**item))
                session.commit()

    def list_shipments(self) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            rows = session.query(ShipmentRecord).all()
            return [
                {col.name: getattr(r, col.name) for col in ShipmentRecord.__table__.columns}
                for r in rows
            ]

    def get_shipment(self, shipment_id: str) -> Optional[Dict[str, Any]]:
        cached = self.cache.get_json(f"shipment:{shipment_id}")
        if cached:
            return cached
        with self.SessionLocal() as session:
            row = session.query(ShipmentRecord).filter_by(shipment_id=shipment_id).first()
            if not row:
                return None
            data = {col.name: getattr(row, col.name) for col in ShipmentRecord.__table__.columns}
            self.cache.set_json(f"shipment:{shipment_id}", data, ttl_seconds=60)
            return data

    def update_shipment_telemetry(self, shipment_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self.SessionLocal() as session:
            row = session.query(ShipmentRecord).filter_by(shipment_id=shipment_id).first()
            if not row:
                return None
            for k, v in updates.items():
                if hasattr(row, k):
                    setattr(row, k, v)
            row.last_updated = datetime.utcnow().isoformat() + "Z"
            session.commit()
            session.refresh(row)
            data = {col.name: getattr(row, col.name) for col in ShipmentRecord.__table__.columns}
            self.cache.set_json(f"shipment:{shipment_id}", data, ttl_seconds=60)
            return data

    def reset_shipments_to_default(self) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            session.query(ShipmentRecord).delete()
            for item in get_live_shipments():
                session.add(ShipmentRecord(**item))
            session.commit()
        return self.list_shipments()

    def log_alert(self, alert_data: Dict[str, Any]) -> None:
        with self.SessionLocal() as session:
            rec = DisruptionAlertRecord(
                shipment_id=alert_data["shipment_id"],
                severity=alert_data.get("severity", "HIGH"),
                delay_probability=alert_data["delay_probability"],
                expected_delay_hours=alert_data["expected_delay_hours"],
                primary_cause=alert_data["primary_cause"],
                recommended_action=alert_data["recommended_action"],
                message=alert_data["alert_message"],
            )
            session.add(rec)
            session.commit()

    def get_recent_alerts(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            rows = (
                session.query(DisruptionAlertRecord)
                .order_by(DisruptionAlertRecord.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "shipment_id": r.shipment_id,
                    "severity": r.severity,
                    "delay_probability": r.delay_probability,
                    "expected_delay_hours": r.expected_delay_hours,
                    "primary_cause": r.primary_cause,
                    "recommended_action": r.recommended_action,
                    "message": r.message,
                    "created_at": r.created_at.isoformat() + "Z" if r.created_at else "",
                }
                for r in rows
            ]

    def get_warehouses(self) -> List[Dict[str, Any]]:
        return [dict(w) for w in WAREHOUSES_CATALOG]

    def get_suppliers(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in SUPPLIERS_CATALOG]


repository = DataRepository()
