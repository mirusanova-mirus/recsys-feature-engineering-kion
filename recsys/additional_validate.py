"""Audit the follow-up against its frozen plan and the unmodified primary run."""
import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from .additional import seal, sha
from .build import load_config
from .experiment import dump
from .metrics import evaluate, paired_bootstrap


def main(config):
    plan = seal(config)
    cfg, rcfg = load_config(config["main_config"]), load_config(config["replication_config"])
    original, out, replica = Path(cfg["output_dir"]), Path(config["output_dir"]), Path(rcfg["output_dir"])
    lock = json.loads((original / "selection.lock.json").read_text())
    record = json.loads((out / "model_records.json").read_text())
    expected = {f"{v}_b{b}_{s}" for v in config["variants"] for b in config["budgets"] for s in config["model_seeds"]}
    assert set(record) == expected
    assert len(record) == 18
    for v in config["variants"]:
        for b in config["budgets"]:
            for s in config["model_seeds"]:
                meta = record[f"{v}_b{b}_{s}"]
                path = Path(meta["model_path"])
                assert sha(path) == meta["model_sha256"]
                model = CatBoostClassifier()
                model.load_model(str(path))
                assert model.feature_names_ == lock["variants"][v]
                params = model.get_all_params()
                assert params["random_seed"] == s and params["iterations"] == b
                assert params["counter_calc_method"] == "SkipTest"
                assert params["depth"] == lock["params"]["depth"]
                assert params["l2_leaf_reg"] == lock["params"]["l2_leaf_reg"]
    metrics = pd.read_csv(out / "metrics.csv")
    boot = json.loads((out / "bootstrap.json").read_text())
    for regime, filename in [("mixed", "test"), ("uniform", "test_uniform")]:
        source = pd.read_parquet(original / "cache" / f"{filename}.parquet")
        qs = {}
        for v in config["variants"]:
            probs = []
            for s in config["model_seeds"]:
                folder = out if v == "all_automatic" else original
                table = pd.read_parquet(folder / "predictions" / f"{regime}_{v}_{s}.parquet")
                assert table[["qid", "user_id", "item_id", "label", "cold_user"]].equals(source[["qid", "user_id", "item_id", "label", "cold_user"]])
                assert np.isfinite(table.score).all() and table.score.between(0,1).all()
                probs.append(table.score.to_numpy())
                if v == "all_automatic" and s == 42:
                    old = pd.read_parquet(original / "predictions" / f"{regime}_{v}_{s}.parquet")
                    assert np.array_equal(old.score.to_numpy(), table.score.to_numpy())
            computed, q = evaluate(source, np.mean(probs, axis=0))
            saved = metrics[(metrics.regime == regime) & (metrics.variant == v) & (metrics.seed == "ensemble")].iloc[0]
            assert abs(saved["ndcg@10"] - computed["ndcg@10"]) < 1e-12
            expected_q = pd.read_csv(out / f"queries_{regime}_{v}.csv")
            pd.testing.assert_frame_equal(q, expected_q, check_exact=False, atol=1e-12)
            qs[v] = q
        recomputed = paired_bootstrap(qs["automatic"], qs["all_automatic"], config["bootstrap_reps"])
        assert recomputed == boot[f"{regime}_bank_vs_selected"]
    budget = pd.read_csv(out / "budget_metrics.csv")
    assert len(budget) == 2 * 2 * 3 * 4  # periods, budgets, variants, seeds + ensemble
    assert not budget.duplicated(["period", "variant", "budget", "seed"]).any()
    bboot = json.loads((out / "budget_bootstrap.json").read_text())
    for period in config["diagnostic_periods"]:
        q = [pd.read_csv(out / f"queries_{period}_{v}_b{b}.csv").sort_values("qid").reset_index(drop=True)
             for v,b in [("all_automatic",450),("automatic",450),("all_automatic",180),("automatic",180)]]
        base, diff = q[0].copy(), q[0].copy()
        base["ndcg@10"] = 0.0
        diff["ndcg@10"] = q[0]["ndcg@10"] - q[1]["ndcg@10"] - q[2]["ndcg@10"] + q[3]["ndcg@10"]
        interaction = paired_bootstrap(base, diff, config["bootstrap_reps"])
        for key in ["delta", "ci95"]:
            assert np.allclose(interaction[key], bboot[f"{period}_budget_interaction"][key], atol=1e-12)
    first, second = [json.loads((folder / "audit.json").read_text()) for folder in [original,replica]]
    a, b = first["splits"]["test"], second["splits"]["test"]
    assert pd.Timestamp(b["end_exclusive"]) <= pd.Timestamp(a["start_inclusive"])
    assert second["dataset"] == first["dataset"]
    assert second["splits"]["test"]["start_inclusive"] == "2021-07-26"
    assert second["splits"]["test"]["end_exclusive"] == "2021-08-02"
    rlock = json.loads((replica / "selection.lock.json").read_text())
    assert rlock["params"] == lock["params"]
    assert rlock["config"] == plan["replication"]
    assert json.loads((replica / "validation.json").read_text())["passed"]
    ecat, ocat = [json.loads((folder / "full_catalog.json").read_text()) for folder in [out,original]]
    assert ecat["chosen_qids"] == ocat["chosen_qids"]
    assert len(ecat["chosen_qids"]) == config["full_catalog_queries"]
    for v in ["manual", "automatic", "popularity_7d"]:
        old, new = [pd.read_csv(folder / f"queries_full_catalog_{v}.csv").sort_values("qid").reset_index(drop=True) for folder in [original,out]]
        pd.testing.assert_frame_equal(old, new, check_exact=False, atol=1e-12)
    result = {"passed": True, "checks": ["original sealed hashes unchanged", "18 factorial cells and exact model hashes",
        "three matching seeds and training caps", "same feature sets, depth, regularization and SkipTest",
        "original bank seed42 predictions unchanged", "three-seed ensembles recomputed", "bootstrap recomputed",
        "budget interaction recomputed", "nonoverlapping temporal test weeks", "same complete user histories",
        "replication parameters frozen", "replication audit passed", "identical original catalog query subset and reference metrics"]}
    dump(out / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/additional.json")
    main(load_config(parser.parse_args().config))
