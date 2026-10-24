"""Preserve shipped measurements before the first fresh reproduction run."""
import json
from pathlib import Path


def prepare(root):
    root = Path(root)
    artifacts = root / "artifacts"
    marker = artifacts / "REFERENCE_RESULTS_ONLY.json"
    if not marker.exists():
        return None
    folders = json.loads(marker.read_text())["folders"]
    assert set(folders) == {"main", "additional", "replication", "independent"}
    number = 1
    snapshot = artifacts / f"reference_snapshot_{number}"
    while snapshot.exists():
        number += 1
        snapshot = artifacts / f"reference_snapshot_{number}"
    snapshot.mkdir()
    for name in folders:
        source = artifacts / name
        if source.exists():
            source.rename(snapshot / name)
    marker.rename(snapshot / marker.name)
    print(f"Preserved supplied measurements in {snapshot}; fresh run can now start.")
    return snapshot


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
