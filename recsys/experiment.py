"""Validation-only finite feature search; seal choices before opening test tables."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd

from .build import load_config
from .metrics import evaluate, paired_bootstrap


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def verify_cache_fingerprint(config):
    out = Path(config["output_dir"])
    fingerprint = {"config": config, "files": {}}
    for path in [out / "cache/train.parquet", out / "cache/search.parquet", out / "feature_registry.json", Path("recsys/features.py")]:
        fingerprint["files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = out / "run_fingerprint.json"
    if path.exists() and json.loads(path.read_text()) != fingerprint:
        raise ValueError("Data, features or config changed; use a fresh output_dir to avoid stale models")
    dump(path, fingerprint)


def reuse_exact_fit(out, target, signature):
    """Reuse an identical fit; never truncate trees or approximate another run."""
    out = Path(out)
    destination = out / "models" / f"{target}.cbm"
    if destination.exists():
        return False
    for path in sorted((out / "models").glob("*.json")):
        meta = json.loads(path.read_text())
        source = path.with_suffix(".cbm")
        if meta.get("signature") == signature and source.exists():
            shutil.copyfile(source, destination)
            meta["original_train_seconds"] = meta.get("original_train_seconds", meta["train_seconds"])
            meta["train_seconds"] = 0.0
            meta["reused_from"] = path.stem
            dump(destination.with_suffix(".json"), meta)
            return True
    return False


def main(config):
    verify_cache_fingerprint(config)
    out = Path(config["output_dir"])
    for folder in ["models", "predictions"]:
        (out / folder).mkdir(exist_ok=True, parents=True)
    registry = json.loads((out / "feature_registry.json").read_text())
    train = pd.read_parquet(out / "cache/train.parquet")
    search = pd.read_parquet(out / "cache/search.parquet")
    confirm = pd.read_parquet(out / "cache/confirm.parquet")
    baseline = registry["baseline"]
    cat = registry["categorical"]
    model_records, search_log = {}, []
    started = time.perf_counter()

    def fit(name, features, params, seed=42, iterations=None):
        iterations = iterations or config["iterations"]
        signature = {"features": features, "params": params, "seed": seed, "iterations": iterations,
                     "learning_rate": config["learning_rate"]}
        model_path = out / "models" / f"{name}.cbm"
        meta_path = model_path.with_suffix(".json")
        reuse_exact_fit(out, name, signature)
        if meta_path.exists() and model_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["signature"] != signature:
                raise ValueError(f"Stale model cache {model_path}; choose a new output directory")
            model = CatBoostClassifier()
            model.load_model(str(model_path))
        else:
            cats = [c for c in cat if c in features]
            tr = Pool(train[features], train.label, cat_features=cats)
            va = Pool(search[features], search.label, cat_features=cats)
            model = CatBoostClassifier(iterations=iterations, learning_rate=config["learning_rate"],
                                       loss_function="Logloss", eval_metric="AUC", random_seed=seed,
                                       thread_count=config["threads"], allow_writing_files=False,
                                       verbose=False, one_hot_max_size=32, counter_calc_method="SkipTest", **params)
            tick = time.perf_counter()
            model.fit(tr, eval_set=va, early_stopping_rounds=50, use_best_model=True)
            seconds = time.perf_counter() - tick
            model.save_model(str(model_path))
            meta = {"signature": signature, "train_seconds": seconds, "trees": model.tree_count_,
                    "feature_count": len(features), "model_bytes": model_path.stat().st_size}
            dump(meta_path, meta)
            del tr, va
        model_records[name] = meta
        return model

    def validation_score(model, features, name, stage):
        pred = model.predict_proba(search[features], thread_count=config["threads"])[:, 1]
        metric, _ = evaluate(search, pred)
        entry = {"name": name, "stage": stage, "features": len(features), **metric}
        search_log.append(entry)
        dump(out / "search_log.json", search_log)
        print(f"{stage} {name}: NDCG@10={metric['ndcg@10']:.6f} AUC={metric['roc_auc']:.6f} "
              f"train={model_records[name]['train_seconds']:.1f}s trees={model.tree_count_}", flush=True)
        return metric["ndcg@10"]

    # Both approaches share the winning baseline hyperparameters.
    hp = []
    for i, params in enumerate(config["grid"]):
        name = f"baseline_grid_{i}"
        model = fit(name, baseline, params)
        hp.append((validation_score(model, baseline, name, "hyperparameters"), i))
        del model
        gc.collect()
    _, best_i = max(hp, key=lambda x: (x[0], -x[1]))
    params = config["grid"][best_i]
    groups = registry["groups"]
    auto_groups = [g for g in groups if g not in ["static", "manual", "recency"]]
    reference = fit("search_reference", baseline, params, iterations=config["search_iterations"])
    reference_score = validation_score(reference, baseline, "search_reference", "feature_screen")
    del reference
    singles = []
    for group in auto_groups:
        feats = baseline + groups[group]
        name = f"screen_{group}"
        model = fit(name, feats, params, iterations=config["search_iterations"])
        score = validation_score(model, feats, name, "feature_screen")
        singles.append((score, group))
        del model
        gc.collect()
    singles.sort(key=lambda x: (-x[0], x[1]))
    top = [g for score, g in singles[:config["max_auto_groups"]]]
    alternatives = [(reference_score, [])] + [(score, [g]) for score, g in singles]
    for n in range(2, len(top) + 1):
        selected = top[:n]
        feats = baseline + sum([groups[g] for g in selected], [])
        name = f"screen_combination_{n}"
        model = fit(name, feats, params, iterations=config["search_iterations"])
        score = validation_score(model, feats, name, "feature_combine")
        alternatives.append((score, selected))
        del model
    best_score, selected = max(alternatives, key=lambda x: (x[0], -len(x[1])))
    if best_score < reference_score + config["min_search_gain"]:
        selected = []
    automatic = baseline + sum([groups[g] for g in selected], [])
    all_auto = baseline + sum([groups[g] for g in auto_groups], [])
    without_indirect = [c for c in baseline if not c.startswith(("user_genre", "user_country", "user_director", "user_type"))]
    variants = {"manual": baseline, "automatic": automatic,
                "all_automatic": all_auto, "without_indirect": without_indirect, "static_only": groups["static"]}
    if "enabled_variants" in config:
        variants = {v: variants[v] for v in config["enabled_variants"]}
    repeated = config.get("repeat_variants", ["manual", "automatic"])
    finals, confirm_log = {}, []
    for variant, feats in variants.items():
        seeds = config["model_seeds"] if variant in repeated else config["model_seeds"][:1]
        for seed in seeds:
            name = f"final_{variant}_{seed}"
            model = fit(name, feats, params, seed)
            pred = model.predict_proba(confirm[feats], thread_count=config["threads"])[:, 1]
            metric, _ = evaluate(confirm, pred)
            confirm_log.append({"variant": variant, "seed": seed, **metric})
            importance = pd.DataFrame({"feature": feats, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False)
            importance.to_csv(out / f"importance_{variant}_{seed}.csv", index=False)
            finals[(variant, seed)] = name
            print(f"confirmation {variant}/{seed}: NDCG@10={metric['ndcg@10']:.6f}", flush=True)
            del model
            gc.collect()
    confirm_scores = {v: float(np.mean([x["ndcg@10"] for x in confirm_log if x["variant"] == v])) for v in variants}
    recommended = "automatic" if confirm_scores["automatic"] > confirm_scores["manual"] else "manual"
    code_hash = hashlib.sha256()
    for f in sorted(Path("recsys").glob("*.py")):
        code_hash.update(f.read_bytes())
    lock = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "selected_groups": selected,
            "params": params, "variants": variants, "model_seeds": config["model_seeds"],
            "recommended_on_confirmation": recommended, "confirmation": confirm_log,
            "primary_comparison": "automatic vs manual, 3-seed probability ensembles, query-macro NDCG@10",
            "secondary_evaluations": ["uniform controls", "cold/warm users", "feature ablations", "full catalog on 210 fixed-seed queries"],
            "source_sha256": code_hash.hexdigest(),
            "audit_sha256": hashlib.sha256((out / "audit.json").read_bytes()).hexdigest(), "config": config}
    # The following file is deliberately saved BEFORE reading any test table.
    dump(out / "selection.lock.json", lock)
    dump(out / "model_records.json", model_records)
    print(f"CHOICES SEALED: groups={selected}, recommended={recommended}; opening test now.", flush=True)
    results, query_results, inference = [], {}, []
    for regime, filename in [("mixed", "test"), ("uniform", "test_uniform")]:
        test = pd.read_parquet(out / "cache" / f"{filename}.parquet")
        for variant, feats in variants.items():
            predictions = []
            seeds = config["model_seeds"] if variant in repeated else config["model_seeds"][:1]
            for seed in seeds:
                name = finals[(variant, seed)]
                model = CatBoostClassifier()
                model.load_model(str(out / "models" / f"{name}.cbm"))
                tick = time.perf_counter()
                p = model.predict_proba(test[feats], thread_count=config["threads"])[:, 1]
                seconds = time.perf_counter() - tick
                inference.append({"regime": regime, "variant": variant, "seed": seed, "seconds": seconds,
                                  "rows": len(test), "seconds_per_100k": seconds / len(test) * 100000})
                predictions.append(p)
                metric, q = evaluate(test, p)
                results.append({"regime": regime, "variant": variant, "seed": str(seed), **metric})
                pred_table = test[["qid", "user_id", "item_id", "label", "cold_user"]].copy()
                pred_table["score"] = p
                pred_table.to_parquet(out / "predictions" / f"{regime}_{variant}_{seed}.parquet", index=False)
                del model
            if variant in repeated:
                mean_p = np.mean(predictions, axis=0)
                metric, q = evaluate(test, mean_p)
                results.append({"regime": regime, "variant": variant, "seed": "ensemble", **metric})
                query_results[(regime, variant)] = q
                q.to_csv(out / f"queries_{regime}_{variant}.csv", index=False)
        # A no-learning baseline, using strictly past popularity.
        for col, name in [("item__count__7d", "popularity_7d"), ("item__count__0d", "popularity_all")]:
            metric, q = evaluate(test, test[col].to_numpy(), probabilities=False)
            results.append({"regime": regime, "variant": name, "seed": "deterministic", **metric})
    comparisons = {regime: paired_bootstrap(query_results[(regime, "manual")], query_results[(regime, "automatic")],
                                           config["bootstrap_reps"]) for regime in ["mixed", "uniform"]}
    if "all_automatic" in repeated:
        for regime in ["mixed", "uniform"]:
            comparisons[f"{regime}_bank_vs_manual"] = paired_bootstrap(
                query_results[(regime, "manual")], query_results[(regime, "all_automatic")], config["bootstrap_reps"])
            comparisons[f"{regime}_bank_vs_selected"] = paired_bootstrap(
                query_results[(regime, "automatic")], query_results[(regime, "all_automatic")], config["bootstrap_reps"])
    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    dump(out / "metrics.json", results)
    dump(out / "bootstrap.json", comparisons)
    dump(out / "inference.json", inference)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "catboost", "scikit-learn", "scipy", "pyarrow", "matplotlib", "requests"]},
                   "threads": config["threads"], "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
                   "experiment_wall_seconds": time.perf_counter() - started,
                   "total_fit_seconds": sum(m["train_seconds"] for m in model_records.values()),
                   "model_count": len(model_records)}
    dump(out / "environment.json", environment)
    print(pd.DataFrame(results)[["regime", "variant", "seed", "ndcg@10", "recall@10", "roc_auc", "logloss"]].to_string(index=False), flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/main.json")
    main(load_config(p.parse_args().config))
