"""Executable audit of the generated, sealed research artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .build import load_config
from .experiment import verify_cache_fingerprint
from .metrics import evaluate


def main(config):
    out = Path(config["output_dir"])
    verify_cache_fingerprint(config)
    audit = json.loads((out / "audit.json").read_text())
    lock = json.loads((out / "selection.lock.json").read_text())
    assert hashlib.sha256((out / "audit.json").read_bytes()).hexdigest() == lock["audit_sha256"]
    assert lock["config"] == config
    for day in audit["days"]:
        assert pd.Timestamp(day["history_max"]) < pd.Timestamp(day["date"])
    previous = None
    forbidden = {"label", "qid", "date", "user_id", "total_dur", "watched_pct", "last_watch_dt", "cold_user"}
    for variant, columns in lock["variants"].items():
        assert not forbidden.intersection(columns), (variant, forbidden.intersection(columns))
        assert len(columns) == len(set(columns))
    for split in ["train", "search", "confirm", "test"]:
        frame = pd.read_parquet(out / "cache" / f"{split}.parquet")
        candidate_hash = hashlib.sha256(frame[["qid", "user_id", "item_id", "label"]].to_numpy().tobytes()).hexdigest()
        assert candidate_hash == audit["splits"][split]["candidate_sha256"]
        assert not frame.duplicated(["qid", "item_id"]).any()
        assert frame.groupby("qid").label.sum().min() >= 1
        expected = config["train_negatives"] if split == "train" else config["eval_negatives"]
        assert ((1 - frame.label).groupby(frame.qid).sum() == expected).all()
        if previous is not None:
            assert previous < frame.date.min()
        previous = frame.date.max()
    results = pd.read_csv(out / "metrics.csv")
    repeated = config.get("repeat_variants", ["manual", "automatic"])
    for variant in repeated:
        probs, table = [], None
        for seed in config["model_seeds"]:
            table = pd.read_parquet(out / "predictions" / f"mixed_{variant}_{seed}.parquet")
            probs.append(table.score.to_numpy())
        score, _ = evaluate(table, np.mean(probs, axis=0))
        saved = results[(results.regime == "mixed") & (results.variant == variant) & (results.seed == "ensemble")].iloc[0]
        assert abs(score["ndcg@10"] - saved["ndcg@10"]) < 1e-12
    full = json.loads((out / "full_catalog.json").read_text())
    assert len(full["chosen_qids"]) == config["full_catalog_queries"]
    resolved = {}
    for variant, columns in lock["variants"].items():
        seeds = config["model_seeds"] if variant in repeated else config["model_seeds"][:1]
        for seed in seeds:
            name = f"final_{variant}_{seed}"
            model = CatBoostClassifier()
            model.load_model(str(out / "models" / f"{name}.cbm"))
            assert model.feature_names_ == columns
            params = model.get_all_params()
            assert params["counter_calc_method"] == "SkipTest"
            assert params["random_seed"] == seed
            importance = pd.read_csv(out / f"importance_{variant}_{seed}.csv")
            assert np.isfinite(importance.importance).all()
            resolved[name] = params
    (out / "catboost_resolved_parameters.json").write_text(json.dumps(resolved, indent=2))
    record = {"passed": True, "checks": ["cache provenance", "strict temporal history", "nonoverlapping splits",
               "no forbidden features", "unique candidates", "negative count", "recomputed ensemble metrics",
               "sealed audit checksum", "full-catalog subset size", "model feature names",
               "SkipTest categorical counters", "model seeds"]}
    (out / "validation.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/main.json")
    main(load_config(p.parse_args().config))
