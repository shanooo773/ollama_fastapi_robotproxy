import json
import time
from pathlib import Path
from typing import Any, Dict


def log_rag_event(log_path: str, event: Dict[str, Any]) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {"timestamp": time.time(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
