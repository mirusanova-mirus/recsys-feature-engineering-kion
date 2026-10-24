"""Measured single-day scaling, independent of model or feature selection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .build import load_config
from .candidates import make_candidates
from .features import FeatureBuilder


def main(config):
    source = Path(config["data_dir"])
    events = pd.read_parquet(source / "events.parquet")
    users, items = pd.read_parquet(source / "users.parquet"), pd.read_parquet(source / "items.parquet")
    ids = np.sort(events.user_id.unique())
    # Nested uniform user samples, no activity filtering and no history truncation.
    ids = np.random.default_rng(2026).permutation(ids)
    day, results = pd.Timestamp("2021-07-31"), []
    for n in [1000, 10000, len(ids)]:
        e = events[events.user_id.isin(ids[:n])]
        b = FeatureBuilder(users, items)
        cstart = time.perf_counter()
        rows, stats = make_candidates(e, day, config["candidate_seed"], config["eval_negatives"], config["uniform_fraction"])
        candidate_seconds = time.perf_counter() - cstart
        tick = time.perf_counter()
        f = b.build(rows, e[e.date < day], day)
        seconds = time.perf_counter() - tick
        results.append({"sample_users": n, "complete_history_rows": len(e), "past_rows": int((e.date < day).sum()),
                        "queries": stats["queries"], "candidate_rows": len(rows), "features": len(sum(b.groups.values(), [])),
                        "candidate_seconds": candidate_seconds, "feature_seconds": seconds,
                        "frame_mib": float(f.memory_usage(deep=True).sum() / 1024**2)})
        print(results[-1], flush=True)
    record = {"date": str(day.date()), "repeats": 1, "kind": "single-day microbenchmark; no extrapolated timings", "results": results}
    (Path(config["output_dir"]) / "scaling.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/main.json")
    main(load_config(p.parse_args().config))
