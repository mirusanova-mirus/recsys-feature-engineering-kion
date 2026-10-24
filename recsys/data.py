"""Download a pinned public KION snapshot; retain complete sampled-user histories."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import pandas as pd
import requests

COMMIT = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
BASE = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{COMMIT}"
FILES = {"data_original.zip": "data_original.zip"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "https://github.com/irsafilo/KION_DATASET", "commit": COMMIT, "files": {}}
    known_path = Path(__file__).resolve().parent.parent / "data/SOURCE_MANIFEST.json"
    known = json.loads(known_path.read_text()) if known_path.exists() else None
    for name, remote in FILES.items():
        path = root / name
        if not path.exists():
            temp = root / (name + ".part")
            for attempt in range(3):
                try:
                    with requests.get(f"{BASE}/{remote}", stream=True, timeout=(30, 120)) as r:
                        r.raise_for_status()
                        with temp.open("wb") as f:
                            for chunk in r.iter_content(1024 * 1024):
                                f.write(chunk)
                    temp.replace(path)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                                   "url": f"{BASE}/{remote}"}
        if known and known["commit"] == COMMIT:
            expected = known["files"][name]["sha256"]
            if manifest["files"][name]["sha256"] != expected:
                raise ValueError(f"Checksum mismatch for {path}; do not use this archive")
        print(f"Verified {name}: {path.stat().st_size:,} bytes", flush=True)
    with zipfile.ZipFile(root / "data_original.zip") as z:
        for basename in ["interactions.csv", "users.csv", "items.csv"]:
            names = [n for n in z.namelist() if Path(n).name == basename]
            if len(names) != 1:
                raise ValueError(f"Unexpected archive members: {z.namelist()}")
            target = root / basename
            with z.open(names[0]) as src, target.open("wb") as out:
                import shutil
                shutil.copyfileobj(src, out)
            manifest["files"][basename] = {"sha256": sha256(target), "bytes": target.stat().st_size}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def prepare(raw, out, n_users=30000, seed=42, exclude_users=None):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Population comes from interactions, not activity-filtered users or future retention.
    all_ids = pd.read_csv(raw / "interactions.csv", usecols=["user_id"])["user_id"].unique()
    all_ids.sort()
    eligible_ids = all_ids
    if exclude_users is not None:
        excluded = pd.read_csv(exclude_users)["user_id"].to_numpy()
        eligible_ids = np.setdiff1d(all_ids, excluded)
        if not len(eligible_ids):
            raise ValueError("No eligible users after exclusion")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(eligible_ids, min(n_users, len(eligible_ids)), replace=False))
    pieces = []
    raw_rows = 0
    for chunk in pd.read_csv(raw / "interactions.csv", chunksize=500000):
        raw_rows += len(chunk)
        pieces.append(chunk[chunk.user_id.isin(selected)])
    original = pd.concat(pieces, ignore_index=True)
    events = original[["user_id", "item_id", "last_watch_dt"]].copy()
    events["date"] = pd.to_datetime(events.pop("last_watch_dt"), errors="raise").dt.normalize()
    duplicates = int(events.duplicated(["user_id", "item_id", "date"]).sum())
    events = events.drop_duplicates().sort_values(["date", "user_id", "item_id"]).reset_index(drop=True)
    users = pd.read_csv(raw / "users.csv")
    users = users[users.user_id.isin(selected)].copy()
    items = pd.read_csv(raw / "items.csv")
    events.to_parquet(out / "events.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    pd.Series(selected, name="user_id").to_csv(out / "sampled_users.csv", index=False)
    stats = {"raw_rows": raw_rows, "population_users": len(all_ids), "sample_users": len(selected),
             "sample_seed": seed, "sample_rows_before_dedup": len(original), "events": len(events),
             "duplicate_triples_removed": duplicates, "items_observed": int(events.item_id.nunique()),
             "date_min": str(events.date.min().date()), "date_max": str(events.date.max().date()),
             "missing_item_metadata": int((~events.item_id.isin(items.item_id)).sum()),
             "history_quantiles": events.groupby("user_id").size().quantile([0, .25, .5, .75, .9, .99, 1]).to_dict(),
             "source_manifest": json.loads((raw / "manifest.json").read_text()),
             "platform": platform.platform()}
    if exclude_users is not None:
        stats.update({"eligible_population_users": len(eligible_ids),
                      "excluded_users": int(len(all_ids)-len(eligible_ids)),
                      "excluded_user_ids_sha256": sha256(exclude_users),
                      "cohort_protocol": "Uniform from users outside the original cohort; no activity filter; retain every provided event."})
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k:v for k,v in stats.items() if k != "source_manifest"}, indent=2), flush=True)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--users", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--exclude-users")
    a = p.parse_args()
    if not a.prepare_only:
        download(a.raw)
    prepare(a.raw, a.out, a.users, a.seed, a.exclude_users)


if __name__ == "__main__":
    main()
