import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, payload: Dict) -> None:
    with path.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


__all__ = ["timestamp", "write_json"]
