"""Shared query construction, with no future-positive filtering of negatives."""
import numpy as np
import pandas as pd


def make_candidates(events, day, seed=42, negatives=50, uniform_fraction=.5, qid_start=0):
    day = pd.Timestamp(day)
    history = events[events.date < day]
    observed = events[events.date == day]
    catalog = np.sort(history.item_id.unique())
    pop = history.groupby("item_id").size().reindex(catalog).to_numpy(dtype=float) ** .75
    if len(catalog):
        pop /= pop.sum()
    by_user = history.groupby("user_id").item_id.agg(set).to_dict()
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(day.value // 86400000000000)]))
    rows = []
    stats = {"raw_targets": len(observed), "unknown_item_targets": 0, "repeated_targets": 0,
             "queries": 0, "cold_user_queries": 0, "eligible_targets": 0}
    for uid, part in observed.groupby("user_id", sort=True):
        seen = by_user.get(uid, set())
        raw = set(part.item_id)
        known = raw.intersection(catalog)
        stats["unknown_item_targets"] += len(raw - known)
        stats["repeated_targets"] += len(known.intersection(seen))
        positives = sorted(known - seen)
        if not positives:
            continue
        # Labels are needed only to exclude today's relevant items from controls.
        # Future user positives remain eligible controls: no oracle filtering.
        eligible = ~np.isin(catalog, list(seen | raw))
        pool = catalog[eligible]
        if len(pool) < negatives:
            continue
        n_uniform = int(round(negatives * uniform_fraction))
        uniform = rng.choice(pool, n_uniform, replace=False)
        remaining = pool[~np.isin(pool, uniform)]
        weights = pop[eligible][~np.isin(pool, uniform)]
        weights /= weights.sum()
        hard = rng.choice(remaining, negatives - n_uniform, replace=False, p=weights)
        candidate_ids = np.concatenate([positives, uniform, hard]).astype(int)
        labels = np.concatenate([np.ones(len(positives)), np.zeros(negatives)]).astype(int)
        order = rng.permutation(len(candidate_ids))
        qid = qid_start + stats["queries"]
        for ix in order:
            rows.append((qid, int(uid), int(candidate_ids[ix]), day, int(labels[ix]), int(not seen)))
        stats["queries"] += 1
        stats["cold_user_queries"] += int(not seen)
        stats["eligible_targets"] += len(positives)
    return pd.DataFrame(rows, columns=["qid", "user_id", "item_id", "date", "label", "cold_user"]), stats
