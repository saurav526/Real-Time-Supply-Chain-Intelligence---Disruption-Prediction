import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_store"
MODELS_DIR = BASE_DIR / "models_store"
MLFLOW_DIR = BASE_DIR / "mlruns"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class Settings:
    app_name: str = "NexusChain AI — Supply Chain Intelligence & Disruption Prediction"
    version: str = "2.4.0"
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{DATA_DIR / 'supply_chain.db'}"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic_telemetry: str = os.getenv("KAFKA_TOPIC_TELEMETRY", "shipment.telemetry.live")
    kafka_topic_alerts: str = os.getenv("KAFKA_TOPIC_ALERTS", "supplychain.disruption.alerts")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"file:///{MLFLOW_DIR.as_posix()}")
    random_seed: int = 42


settings = Settings()
