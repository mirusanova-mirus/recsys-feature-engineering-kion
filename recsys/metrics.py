"""Binary sampled-candidate metrics and paired user-cluster bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def query_metrics(frame, scores):
    df = frame[["qid", "user_id", "item_id", "label", "cold_user"]].copy()
    df["score"] = np.asarray(scores)
    # Stable ties, independent of input order and label placement.
    df = df.sort_values(["qid", "score", "item_id"], ascending=[True, False, True])
    df["rank"] = df.groupby("qid").cumcount() + 1
    out = df.groupby("qid").agg(user_id=("user_id", "first"), cold_user=("cold_user", "first"), positives=("label", "sum"))
    for k in [5, 10, 20]:
        top = df[df["rank"] <= k].copy()
        top["gain"] = top.label / np.log2(top["rank"] + 1)
        dcg = top.groupby("qid").gain.sum().reindex(out.index, fill_value=0)
        idcg = out.positives.map(lambda n: np.sum(1 / np.log2(np.arange(1, min(k, n) + 1) + 1)))
        out[f"ndcg@{k}"] = dcg / idcg
        out[f"recall@{k}"] = top.groupby("qid").label.sum().reindex(out.index, fill_value=0) / out.positives
    first = df[df.label == 1].groupby("qid")["rank"].min()
    out["mrr"] = 1 / first
    return out.reset_index()


def evaluate(frame, scores, probabilities=True):
    q = query_metrics(frame, scores)
    m = {c: float(q[c].mean()) for c in q if c.startswith(("ndcg@", "recall@")) or c == "mrr"}
    m["roc_auc"] = float(roc_auc_score(frame.label, scores))
    m["logloss"] = float(log_loss(frame.label, np.clip(scores, 1e-7, 1-1e-7))) if probabilities else None
    m["queries"] = len(q)
    for cold, name in [(0, "warm"), (1, "cold")]:
        sub = q[q.cold_user == cold]
        m[f"{name}_queries"] = len(sub)
        m[f"{name}_ndcg@10"] = float(sub["ndcg@10"].mean()) if len(sub) else None
    return m, q


def paired_bootstrap(base, challenger, reps=2000, seed=2026):
    keys = ["qid", "user_id"]
    a = base[keys].sort_values(keys).reset_index(drop=True)
    b = challenger[keys].sort_values(keys).reset_index(drop=True)
    if not a.equals(b):
        raise ValueError("Paired bootstrap requires identical queries and users")
    merged = base.merge(challenger, on=["qid", "user_id"], suffixes=("_base", "_auto"), validate="one_to_one")
    delta = merged["ndcg@10_auto"] - merged["ndcg@10_base"]
    tmp = pd.DataFrame({"user": merged.user_id, "delta": delta, "one": 1})
    agg = tmp.groupby("user").agg(delta=("delta", "sum"), n=("one", "sum"))
    rng = np.random.default_rng(seed)
    values = []
    # Resample users, retain all of each user's queries, and keep query-macro estimand.
    for _ in range(reps):
        ix = rng.integers(0, len(agg), len(agg))
        values.append(float(agg.delta.to_numpy()[ix].sum() / agg.n.to_numpy()[ix].sum()))
    low, high = np.quantile(values, [.025, .975])
    return {"metric": "ndcg@10", "estimand": "query_macro_difference", "cluster": "user_id",
            "delta": float(delta.mean()), "ci95": [float(low), float(high)],
            "bootstrap_replicates": reps, "users": len(agg), "seed": seed}
