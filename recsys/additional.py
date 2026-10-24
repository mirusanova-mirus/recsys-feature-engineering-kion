"""Locked, explicitly post-hoc extensions; preserve the original test and selection."""
import argparse
import gc
import hashlib
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
from .experiment import dump, verify_cache_fingerprint
from .features import FeatureBuilder
from .metrics import evaluate, paired_bootstrap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(config):
    """Freeze both follow-up designs before running either new experiment."""
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    main_config = load_config(config["main_config"])
    main = Path(main_config["output_dir"])
    verify_cache_fingerprint(main_config)
    paths = [main / n for n in ["selection.lock.json", "metrics.csv", "metrics.json", "bootstrap.json", "full_catalog.json",
                                "audit.json", "feature_registry.json", "run_fingerprint.json"]]
    paths += [main / "cache" / f"{s}.parquet" for s in ["train", "search", "confirm", "test", "test_uniform"]]
    paths += [Path(config["main_config"]), Path(config["replication_config"])]
    fingerprint = {"config": config, "inputs": {str(p): sha(p) for p in paths}}
    destination = out / "plan.lock.json"
    if destination.exists():
        plan = json.loads(destination.read_text())
        if plan["fingerprint"] != fingerprint:
            raise ValueError("Supplement inputs changed; use a new output_dir")
        return plan
    plan = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "Post-hoc extension: original mixed, uniform and catalog test metrics were already seen.",
        "fingerprint": fingerprint,
        "repeats": "Repeat the already-defined full-bank variant with seeds 42,43,44; no change to the primary comparison.",
        "budget_control": "Fixed original manual/selected/full-bank sets x caps 180/450 x three seeds; score search and confirm only. No new feature choice from these outcomes.",
        "budget_interaction": "On confirm, (bank450-selected450)-(bank180-selected180), paired user-cluster bootstrap. No other feature search added.",
        "replication": load_config(config["replication_config"]),
        "replication_policy": "Re-run the original selector on earlier train/search; freeze depth=6,l2=3, all other settings. Three seeds for manual, selected and bank. Seal selection before test 2021-07-26..2021-08-01.",
        "temporal_independence_limit": "Test weeks do not overlap, but the same users/source are reused and earlier outcomes were part of the original training history. Retrospective temporal replication, not a pristine independently collected holdout.",
        "no_catalog_tuning": "Evaluate raw scores plus fixed seven-day popularity on the same 210 catalog queries; do not correct sampling or choose a model from catalog test outcomes.",
    }
    dump(destination, plan)
    print(f"Supplement plan sealed: {destination}", flush=True)
    return plan


def run(config):
    plan = seal(config)
    cfg = load_config(config["main_config"])
    source, out = Path(cfg["output_dir"]), Path(config["output_dir"])
    lock = json.loads((source / "selection.lock.json").read_text())
    registry = json.loads((source / "feature_registry.json").read_text())
    for folder in ["models", "predictions"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(source / "cache/train.parquet")
    search = pd.read_parquet(source / "cache/search.parquet")
    confirm = pd.read_parquet(source / "cache/confirm.parquet")
    params, records, paths = lock["params"], {}, {}
    started = time.perf_counter()

    def fit(variant, budget, seed):
        columns = lock["variants"][variant]
        signature = {"features": columns, "params": params, "seed": seed, "iterations": budget,
                     "learning_rate": cfg["learning_rate"]}
        name = f"{variant}_b{budget}_{seed}"
        target = out / "models" / f"{name}.cbm"
        meta_path = target.with_suffix(".json")
        if target.exists():
            meta = json.loads(meta_path.read_text())
            if meta["signature"] != signature:
                raise ValueError(f"Stale supplement model: {target}")
            model_path, fresh = target, False
        else:
            match = None
            for candidate in sorted((source / "models").glob("*.json")):
                meta = json.loads(candidate.read_text())
                if meta.get("signature") == signature and candidate.with_suffix(".cbm").exists():
                    match = candidate
                    break
            if match:
                meta = json.loads(match.read_text())
                meta = {**meta, "reused_from": str(match.with_suffix(".cbm")),
                        "original_train_seconds": meta.get("original_train_seconds", meta["train_seconds"]),
                        "train_seconds": 0.0}
                model_path, fresh = match.with_suffix(".cbm"), False
            else:
                model = CatBoostClassifier(iterations=budget, learning_rate=cfg["learning_rate"],
                    loss_function="Logloss", eval_metric="AUC", random_seed=seed, thread_count=cfg["threads"],
                    allow_writing_files=False, verbose=False, one_hot_max_size=32,
                    counter_calc_method="SkipTest", **params)
                cats = [c for c in registry["categorical"] if c in columns]
                tr = Pool(train[columns], train.label, cat_features=cats)
                va = Pool(search[columns], search.label, cat_features=cats)
                tick = time.perf_counter()
                model.fit(tr, eval_set=va, early_stopping_rounds=50, use_best_model=True)
                seconds = time.perf_counter() - tick
                model.save_model(str(target))
                meta = {"signature": signature, "train_seconds": seconds, "trees": model.tree_count_,
                        "feature_count": len(columns), "model_bytes": target.stat().st_size}
                dump(meta_path, meta)
                model_path, fresh = target, True
                del tr, va
        if not fresh:
            model = CatBoostClassifier()
            model.load_model(str(model_path))
        assert model.feature_names_ == columns
        assert model.get_all_params()["counter_calc_method"] == "SkipTest"
        records[name] = {**meta, "model_path": str(model_path), "model_sha256": sha(model_path)}
        paths[(variant, budget, seed)] = model_path
        dump(out / "model_records.json", records)
        print(f"additional {name}: fit={meta['train_seconds']:.1f}s, trees={model.tree_count_}", flush=True)
        return model

    diagnostic, q_by_key = [], {}
    # A factorial control, without searching for a new winning feature subset.
    for budget in config["budgets"]:
        for variant in config["variants"]:
            predictions = {"search": [], "confirm": []}
            columns = lock["variants"][variant]
            for seed in config["model_seeds"]:
                model = fit(variant, budget, seed)
                for period, table in [("search", search), ("confirm", confirm)]:
                    p = model.predict_proba(table[columns], thread_count=cfg["threads"])[:, 1]
                    predictions[period].append(p)
                    metric, _ = evaluate(table, p)
                    diagnostic.append({"period": period, "variant": variant, "budget": budget,
                                       "seed": str(seed), **metric})
                del model
                gc.collect()
            for period, table in [("search", search), ("confirm", confirm)]:
                metric, q = evaluate(table, np.mean(predictions[period], axis=0))
                diagnostic.append({"period": period, "variant": variant, "budget": budget,
                                   "seed": "ensemble", **metric})
                q_by_key[(period, variant, budget)] = q
                q.to_csv(out / f"queries_{period}_{variant}_b{budget}.csv", index=False)
            pd.DataFrame(diagnostic).to_csv(out / "budget_metrics.csv", index=False)
    comparisons = {}
    for period in config["diagnostic_periods"]:
        for budget in config["budgets"]:
            comparisons[f"{period}_bank_vs_selected_b{budget}"] = paired_bootstrap(
                q_by_key[(period, "automatic", budget)], q_by_key[(period, "all_automatic", budget)], config["bootstrap_reps"])
        aligned = [q_by_key[(period, v, b)].sort_values("qid").reset_index(drop=True)
                   for v, b in [("all_automatic", 450), ("automatic", 450), ("all_automatic", 180), ("automatic", 180)]]
        for q in aligned[1:]:
            assert q[["qid", "user_id"]].equals(aligned[0][["qid", "user_id"]])
        base, difference = aligned[0].copy(), aligned[0].copy()
        base["ndcg@10"] = 0.0
        difference["ndcg@10"] = (aligned[0]["ndcg@10"] - aligned[1]["ndcg@10"]) - (aligned[2]["ndcg@10"] - aligned[3]["ndcg@10"])
        interaction = paired_bootstrap(base, difference, config["bootstrap_reps"])
        interaction["interpretation"] = "(bank450-selected450)-(bank180-selected180)"
        comparisons[f"{period}_budget_interaction"] = interaction
    dump(out / "budget_bootstrap.json", comparisons)
    print(pd.DataFrame(diagnostic).query("seed == 'ensemble'")[["period", "variant", "budget", "ndcg@10"]].to_string(index=False), flush=True)
    del train, search, confirm
    gc.collect()

    results, boot = [], {}
    for regime, filename in [("mixed", "test"), ("uniform", "test_uniform")]:
        table = pd.read_parquet(source / "cache" / f"{filename}.parquet")
        query_results = {}
        for variant in config["variants"]:
            predictions = []
            for seed in config["model_seeds"]:
                if variant in ["manual", "automatic"]:
                    old = pd.read_parquet(source / "predictions" / f"{regime}_{variant}_{seed}.parquet")
                    assert old[["qid", "user_id", "item_id", "label", "cold_user"]].equals(table[["qid", "user_id", "item_id", "label", "cold_user"]])
                    p = old.score.to_numpy()
                else:
                    model = CatBoostClassifier()
                    model.load_model(str(paths[(variant, 450, seed)]))
                    p = model.predict_proba(table[lock["variants"][variant]], thread_count=cfg["threads"])[:, 1]
                    saved = table[["qid", "user_id", "item_id", "label", "cold_user"]].copy()
                    saved["score"] = p
                    saved.to_parquet(out / "predictions" / f"{regime}_{variant}_{seed}.parquet", index=False)
                    if seed == 42:
                        original = pd.read_parquet(source / "predictions" / f"{regime}_{variant}_{seed}.parquet")
                        assert np.array_equal(p, original.score.to_numpy()), "Original seed42 predictions changed"
                    del model
                predictions.append(p)
                metric, _ = evaluate(table, p)
                results.append({"regime": regime, "variant": variant, "seed": str(seed), **metric})
            metric, q = evaluate(table, np.mean(predictions, axis=0))
            results.append({"regime": regime, "variant": variant, "seed": "ensemble", **metric})
            query_results[variant] = q
            q.to_csv(out / f"queries_{regime}_{variant}.csv", index=False)
        for base, challenger, name in [("manual", "automatic", "selected_vs_manual"),
                                      ("manual", "all_automatic", "bank_vs_manual"),
                                      ("automatic", "all_automatic", "bank_vs_selected")]:
            boot[f"{regime}_{name}"] = paired_bootstrap(query_results[base], query_results[challenger], config["bootstrap_reps"])
        del table
        gc.collect()
    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    dump(out / "metrics.json", results)
    dump(out / "bootstrap.json", boot)
    catalog(config, cfg, lock, paths)
    # Re-read the sealed inputs and reject any silent change to the original experiment.
    seal(config)
    dump(out / "environment.json", {"platform": platform.platform(), "wall_seconds": time.perf_counter()-started,
        "total_new_fit_seconds": sum(m["train_seconds"] for m in records.values()),
        "logical_fits": len(records), "new_fits": sum(m["train_seconds"] > 0 for m in records.values()),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
        "schedule": "One training process for supplements; the separate temporal replication may run concurrently. Thread count per fit 6."})
    print("Additional experiments complete; original primary artifacts preserved.", flush=True)


def catalog(config, cfg, lock, paths):
    """Same primary catalog query sample, with all three full-bank seeds added."""
    source, out = Path(cfg["output_dir"]), Path(config["output_dir"])
    original = json.loads((source / "full_catalog.json").read_text())
    test = pd.read_parquet(source / "cache/test.parquet")
    qids = np.sort(test.qid.unique())
    chosen = np.random.default_rng(config["catalog_seed"]).choice(qids, config["full_catalog_queries"], replace=False)
    assert sorted(map(int, chosen)) == original["chosen_qids"]
    queries = test[test.qid.isin(chosen)]
    data = Path(cfg["data_dir"])
    events = pd.read_parquet(data / "events.parquet")
    builder = FeatureBuilder(pd.read_parquet(data / "users.parquet"), pd.read_parquet(data / "items.parquet"))
    models = {}
    for variant in config["variants"]:
        models[variant] = []
        for seed in config["model_seeds"]:
            model = CatBoostClassifier()
            model.load_model(str(paths[(variant, 450, seed)]))
            models[variant].append(model)
    all_q = {v: [] for v in [*models, "popularity_7d"]}
    counts, started = [], time.perf_counter()
    for day, part in queries.groupby("date", sort=True):
        history = events[events.date < day]
        catalog_items = np.sort(history.item_id.unique())
        seen = history.groupby("user_id").item_id.agg(set).to_dict()
        pieces = []
        for qid, q in part.groupby("qid", sort=True):
            uid = int(q.user_id.iloc[0])
            candidates = np.setdiff1d(catalog_items, list(seen.get(uid, set())))
            positives = q[q.label == 1].item_id.to_numpy()
            pieces.append(pd.DataFrame({"qid": qid, "user_id": uid, "item_id": candidates,
                "date": day, "label": np.isin(candidates, positives).astype(int), "cold_user": int(q.cold_user.iloc[0])}))
        frame = builder.build(pd.concat(pieces, ignore_index=True), history, day)
        for variant, fitted in models.items():
            p = np.mean([m.predict_proba(frame[lock["variants"][variant]], thread_count=cfg["threads"])[:, 1] for m in fitted], axis=0)
            _, q = evaluate(frame, p)
            all_q[variant].append(q)
        _, q = evaluate(frame, frame["item__count__7d"], probabilities=False)
        all_q["popularity_7d"].append(q)
        counts.append({"date": str(day.date()), "rows": len(frame), "queries": len(pieces)})
        print(f"Additional catalog {day.date()}: {len(frame):,} candidates", flush=True)
    combined, results = {}, []
    for variant, pieces in all_q.items():
        q = pd.concat(pieces, ignore_index=True)
        combined[variant] = q
        q.to_csv(out / f"queries_full_catalog_{variant}.csv", index=False)
        values = {c: float(q[c].mean()) for c in q if c.startswith(("ndcg@", "recall@")) or c == "mrr"}
        results.append({"variant": variant, "queries": len(q), **values})
        if variant in ["manual", "automatic", "popularity_7d"]:
            old = pd.read_csv(source / f"queries_full_catalog_{variant}.csv").sort_values("qid").reset_index(drop=True)
            pd.testing.assert_frame_equal(q.sort_values("qid").reset_index(drop=True), old, check_exact=False, atol=1e-12)
    comparisons = {}
    for a, b, name in [("manual", "all_automatic", "bank_vs_manual"), ("automatic", "all_automatic", "bank_vs_selected"),
                       ("popularity_7d", "all_automatic", "bank_vs_popularity")]:
        comparisons[name] = paired_bootstrap(combined[a], combined[b], config["bootstrap_reps"])
    dump(out / "full_catalog.json", {"chosen_qids": sorted(map(int, chosen)), "days": counts,
        "results": results, "comparisons": comparisons, "wall_seconds": time.perf_counter()-started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/additional.json")
    parser.add_argument("--seal-only", action="store_true")
    parser.add_argument("--renew-plan", action="store_true", help="Archive generated follow-up outputs before a full new reproduction")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.renew_plan:
        replica_cfg = load_config(cfg["replication_config"])
        archive = Path("artifacts/archive") / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
        previous_outputs = [Path(cfg["output_dir"]), Path(replica_cfg["output_dir"])]
        holdout_config = Path("configs/user_holdout.json")
        if holdout_config.exists():
            previous_outputs.append(Path(load_config(holdout_config)["output_dir"]))
        for path in previous_outputs:
            if path.exists():
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(archive / path.name))
                print(f"Previous generated run archived: {path} -> {archive / path.name}", flush=True)
    seal(cfg) if args.seal_only else run(cfg)
