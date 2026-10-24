"""User-disjoint follow-up, sealed before sampling the previously unused cohort."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .additional import sha
from .build import build, load_config
from .catalog import main as evaluate_catalog
from .data import prepare
from .experiment import dump, main as experiment
from .validate import main as validate


def audit(config):
    out, source = Path(config["output_dir"]), Path(config["data_dir"])
    plan = json.loads((out / "plan.lock.json").read_text())
    assert plan["config"] == config
    assert plan["excluded_users_sha256"] == sha(config["exclude_users"])
    selected = pd.read_csv(source / "sampled_users.csv").user_id.to_numpy()
    original = pd.read_csv(config["exclude_users"]).user_id.to_numpy()
    assert len(selected) == config["cohort_users"]
    assert not len(np.intersect1d(selected, original))
    # Verify complete user histories directly against the pinned raw CSV.
    expected = []
    for chunk in pd.read_csv("data/raw/interactions.csv", usecols=["user_id","item_id","last_watch_dt"], chunksize=500000):
        expected.append(chunk[chunk.user_id.isin(selected)])
    expected = pd.concat(expected, ignore_index=True).rename(columns={"last_watch_dt":"date"})
    expected["date"] = pd.to_datetime(expected.date).dt.normalize()
    expected = expected.drop_duplicates().sort_values(["date","user_id","item_id"]).reset_index(drop=True)
    actual = pd.read_parquet(source / "events.parquet")
    pd.testing.assert_frame_equal(expected[["user_id","item_id","date"]], actual)
    stats = json.loads((source / "stats.json").read_text())
    assert stats["sample_seed"] == config["cohort_seed"]
    rng = np.random.default_rng(config["cohort_seed"])
    population = np.sort(pd.read_csv("data/raw/interactions.csv", usecols=["user_id"]).user_id.unique())
    eligible = np.setdiff1d(population, original)
    np.testing.assert_array_equal(selected, np.sort(rng.choice(eligible, config["cohort_users"], replace=False)))
    assert json.loads((out / "validation.json").read_text())["passed"]
    record = {"passed": True, "users": len(selected), "user_overlap_with_primary": 0,
              "complete_histories_verified": True, "uniform_user_sample_recomputed": True,
              "events": len(actual), "selected_users_sha256": sha(source / "sampled_users.csv"),
              "plan_created_utc": plan["created_utc"],
              "scope": "Independent users within the same KION source; shared items/metadata and calendar mean this is not a new environment."}
    dump(out / "cohort_validation.json", record)
    print(json.dumps(record, indent=2), flush=True)


def main(config):
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    plan = {"config": config, "excluded_users_sha256": sha(config["exclude_users"]),
            "source_manifest_sha256": sha("data/SOURCE_MANIFEST.json"),
            "sampling_policy": "Uniform 5000 users from the complement of the initial 30000; no event/activity screening.",
            "evaluation_policy": "Original temporal split, new-cohort histories only, full selector repeated, three seeds per variant, frozen hyperparameters transferred from the original cohort. Catalog is secondary, no tuning on test.",
            "comparison": "automatic vs manual, and bank vs manual/automatic, user-cluster bootstrap; all are supplemental, not replacements for the original primary test."}
    target = out / "plan.lock.json"
    if target.exists():
        saved = json.loads(target.read_text())
        assert {k:v for k,v in saved.items() if k != "created_utc"} == plan
    else:
        dump(target, {**plan, "created_utc": pd.Timestamp.now(tz="UTC").isoformat()})
        print(f"User-disjoint plan sealed BEFORE sampling: {target}", flush=True)
    started = time.perf_counter()
    prepare("data/raw", config["data_dir"], config["cohort_users"], config["cohort_seed"], config["exclude_users"])
    build(config)
    experiment(config)
    evaluate_catalog(config)
    validate(config)
    audit(config)
    dump(out / "followup_runtime.json", {"wall_seconds": time.perf_counter()-started,
        "schedule": "Two threads per fit; may overlap the separate six-thread temporal replication or main-bank follow-up. No aggregate process RSS measured."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/user_holdout.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    audit(config) if args.audit_only else main(config)
