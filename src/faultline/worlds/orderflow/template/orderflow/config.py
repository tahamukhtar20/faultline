"""Service settings."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ORDERFLOW_DATA", "/data"))
DATABASE_PATH = DATA_DIR / "orders.db"
PROVIDER_URL = os.environ.get("ORDERFLOW_PROVIDER_URL", "http://provider:8100")

LEASE_SECONDS = 3.0
POLL_SECONDS = 0.05
PROVIDER_TIMEOUT_SECONDS = 2.5
MAX_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 0.1
