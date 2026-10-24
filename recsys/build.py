import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from .candidates import make_candidates
from .features import FeatureBuilder


def load_config(path):
    return json.loads(Path(path).read_text())


def split_bounds(events, config):
    end = pd.Timestamp(config.get("end_exclusive", events.date.max() + pd.Timedelta(days=1)))
    if end > events.date.max() + pd.Timedelta(days=1):
        raise ValueError("Requested evaluation end extends beyond the available events")
    if any(config[key] <= 0 for key in ["test_days", "confirmation_days", "search_days", "train_days"]):
        raise ValueError("All temporal split lengths must be positive")
    test_start = end - pd.Timedelta(days=config["test_days"])
    confirm_start = test_start - pd.Timedelta(days=config["confirmation_days"])
    search_start = confirm_start - pd.Timedelta(days=config["search_days"])
    train_start = search_start - pd.Timedelta(days=config["train_days"])
    return {"train": (train_start, search_start), "search": (search_start, confirm_start),
            "confirm": (confirm_start, test_start), "test": (test_start, end)}


def build(config):
    source, out = Path(config["data_dir"]), Path(config["output_dir"])
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(source / "events.parquet")
    builder = FeatureBuilder(pd.read_parquet(source / "users.parquet"), pd.read_parquet(source / "items.parquet"))
    bounds = split_bounds(events, config)
    audit = {"splits": {}, "days": [], "strict_history": True,
             "protocol": "Daily prequential: events at t may update features only for dates > t; weights remain fixed.",
             "candidate_protocol": "All eligible daily positives + unique controls, 50% uniform / 50% past popularity^.75; no future-positive exclusion.",
             "config": config, "dataset": json.loads((source / "stats.json").read_text())}
    qid, started = 0, time.perf_counter()
    for split, (start, stop) in bounds.items():
        pieces, digest, secondary = [], hashlib.sha256(), []
        split_start = time.perf_counter()
        for day in pd.date_range(start, stop - pd.Timedelta(days=1)):
            nneg = config["train_negatives"] if split == "train" else config["eval_negatives"]
            rows, stats = make_candidates(events, day, config["candidate_seed"], nneg, config["uniform_fraction"], qid)
            qid += stats["queries"]
            history = events[events.date < day]
            assert history.date.max() < day
            feat = builder.build(rows, history, day)
            pieces.append(feat)
            digest.update(rows[["qid", "user_id", "item_id", "label"]].to_numpy().tobytes())
            record = {"split": split, "date": str(day.date()), "history_max": str(history.date.max().date()),
                      "history_rows": len(history), "rows": len(rows), **stats}
            audit["days"].append(record)
            if split == "test":
                # Predeclared sensitivity evaluation: all controls uniform.
                alt, _ = make_candidates(events, day, config["candidate_seed"], nneg, 1.0,
                                         qid - stats["queries"])
                secondary.append(builder.build(alt, history, day))
            print(f"{split} {day.date()} queries={stats['queries']} rows={len(rows)}", flush=True)
        table = pd.concat(pieces, ignore_index=True)
        table.to_parquet(cache / f"{split}.parquet", index=False)
        elapsed = time.perf_counter() - split_start
        audit["splits"][split] = {"start_inclusive": str(start.date()), "end_exclusive": str(stop.date()),
                                    "rows": len(table), "queries": int(table.qid.nunique()),
                                    "users": int(table.user_id.nunique()), "positive_rows": int(table.label.sum()),
                                    "candidate_sha256": digest.hexdigest(), "build_seconds": elapsed,
                                    "file_bytes": (cache / f"{split}.parquet").stat().st_size}
        if secondary:
            pd.concat(secondary, ignore_index=True).to_parquet(cache / "test_uniform.parquet", index=False)
    audit["build_seconds"] = time.perf_counter() - started
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    (out / "feature_registry.json").write_text(json.dumps(builder.registry(), indent=2, ensure_ascii=False))
    print(f"Built in {audit['build_seconds']:.1f}s", flush=True)
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/main.json")
    build(load_config(p.parse_args().config))
