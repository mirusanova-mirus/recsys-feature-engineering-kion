"""Predeclared full available-catalog sensitivity check on a random query subset."""
import argparse
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from .build import load_config
from .features import FeatureBuilder
from .metrics import evaluate, paired_bootstrap


def main(config):
    out, source = Path(config["output_dir"]), Path(config["data_dir"])
    lock = json.loads((out / "selection.lock.json").read_text())
    if lock["config"] != config:
        raise ValueError("Full-catalog evaluation must use the sealed experiment config")
    test = pd.read_parquet(out / "cache/test.parquet")
    qids = np.sort(test.qid.unique())
    chosen = np.random.default_rng(2026).choice(qids, min(config["full_catalog_queries"], len(qids)), replace=False)
    queries = test[test.qid.isin(chosen)]
    events = pd.read_parquet(source / "events.parquet")
    builder = FeatureBuilder(pd.read_parquet(source / "users.parquet"), pd.read_parquet(source / "items.parquet"))
    models = {}
    for variant in config.get("catalog_variants", ["manual", "automatic"]):
        models[variant] = []
        for seed in config["model_seeds"]:
            model = CatBoostClassifier()
            model.load_model(str(out / "models" / f"final_{variant}_{seed}.cbm"))
            models[variant].append(model)
    all_q = {v: [] for v in models}
    all_q["popularity_7d"] = []
    counts, started, build_seconds, infer_seconds = [], time.perf_counter(), 0, 0
    for day, part in queries.groupby("date", sort=True):
        history = events[events.date < day]
        catalog = np.sort(history.item_id.unique())
        seen = history.groupby("user_id").item_id.agg(set).to_dict()
        pieces = []
        for qid, q in part.groupby("qid", sort=True):
            uid = int(q.user_id.iloc[0])
            candidates = np.setdiff1d(catalog, list(seen.get(uid, set())))
            positives = q[q.label == 1].item_id.to_numpy()
            pieces.append(pd.DataFrame({"qid": qid, "user_id": uid, "item_id": candidates,
                                       "date": day, "label": np.isin(candidates, positives).astype(int),
                                       "cold_user": int(q.cold_user.iloc[0])}))
        rows = pd.concat(pieces, ignore_index=True)
        tick = time.perf_counter()
        f = builder.build(rows, history, day)
        build_seconds += time.perf_counter() - tick
        for variant, fitted in models.items():
            tick = time.perf_counter()
            scores = np.mean([model.predict_proba(f[lock["variants"][variant]], thread_count=config["threads"])[:, 1]
                              for model in fitted], axis=0)
            infer_seconds += time.perf_counter() - tick
            _, qm = evaluate(f, scores)
            all_q[variant].append(qm)
        _, pop_q = evaluate(f, f["item__count__7d"], probabilities=False)
        all_q["popularity_7d"].append(pop_q)
        counts.append({"date": str(day.date()), "queries": len(pieces), "rows": len(f), "catalog_items": len(catalog)})
        print(f"Full catalog {day.date()}: {len(pieces)} queries, {len(f):,} candidates", flush=True)
    combined = {v: pd.concat(q, ignore_index=True) for v, q in all_q.items()}
    results = []
    for variant, q in combined.items():
        q.to_csv(out / f"queries_full_catalog_{variant}.csv", index=False)
        values = {c: float(q[c].mean()) for c in q if c.startswith(("ndcg@", "recall@")) or c == "mrr"}
        results.append({"variant": variant, "queries": len(q), **values})
    boot = paired_bootstrap(combined["manual"], combined["automatic"], config["bootstrap_reps"])
    comparisons = {"automatic_vs_manual": boot}
    if "all_automatic" in combined:
        comparisons["bank_vs_manual"] = paired_bootstrap(combined["manual"], combined["all_automatic"], config["bootstrap_reps"])
        comparisons["bank_vs_selected"] = paired_bootstrap(combined["automatic"], combined["all_automatic"], config["bootstrap_reps"])
    for variant in models:
        comparisons[f"{variant}_vs_popularity"] = paired_bootstrap(combined["popularity_7d"], combined[variant], config["bootstrap_reps"])
    record = {"sampling": "210 queries uniformly without replacement from eligible test queries; seed 2026",
              "chosen_qids": sorted(map(int, chosen)), "days": counts, "results": results, "bootstrap": boot,
              "comparisons": comparisons,
              "build_seconds": build_seconds, "inference_seconds": infer_seconds,
              "inference_model_count": sum(map(len, models.values())),
              "wall_seconds": time.perf_counter() - started}
    (out / "full_catalog.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/main.json")
    main(load_config(p.parse_args().config))
