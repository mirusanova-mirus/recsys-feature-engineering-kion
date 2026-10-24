"""A finite typed grammar of temporal aggregates. All windows are [t-w, t)."""
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    relation: str
    operation: str
    window_days: Optional[int] = None
    half_life: Optional[int] = None


RELATIONS = {
    "user": ["user_id"], "item": ["item_id"],
    "user_genre": ["user_id", "genre"],
    "user_country": ["user_id", "country"],
    "user_director": ["user_id", "director"],
    "user_type": ["user_id", "content_type"],
    "segment_item": ["age", "item_id"],
}
CORE = list(RELATIONS)[:6]


def grammar():
    specs = []
    for w in [7, 30, 0, 1, 3, 10, 14, 60]:
        group = "manual" if w in [7, 30, 0] else f"window_{w}d"
        for rel in CORE:
            specs.append(FeatureSpec(f"{rel}__count__{w}d", group, rel, "count", w))
            if rel.startswith("user_"):
                specs.append(FeatureSpec(f"{rel}__share__{w}d", group, rel, "share", w))
    for half in [2, 10, 30]:
        for rel in CORE:
            specs.append(FeatureSpec(f"{rel}__ewm__h{half}", f"decay_{half}d", rel, "ewm", half_life=half))
    for w in [7, 30, 0]:
        specs.append(FeatureSpec(f"segment_item__count__{w}d", "segment", "segment_item", "count", w))
    return specs


def first_token(series):
    return series.fillna("__MISSING__").astype(str).str.split(",").str[0].str.strip().replace("", "__MISSING__")


class FeatureBuilder:
    def __init__(self, users, items):
        self.users = users.set_index("user_id").copy()
        self.items = items.set_index("item_id").copy()
        # The full token list is additionally used in multi-genre affinities.
        self.genre_tokens = self.items.genres.fillna("").astype(str).map(
            lambda x: tuple(sorted(set(t.strip() for t in x.split(",") if t.strip()))))
        for dest, source in [("genre", "genres"), ("country", "countries"), ("director", "directors")]:
            self.items[dest] = first_token(self.items[source])
        self.user_cols = [c for c in ["sex", "age", "income", "kids_flg"] if c in self.users]
        self.item_cols = [c for c in ["content_type", "genre", "country", "director", "age_rating", "for_kids"] if c in self.items]
        self.cat_cols = self.user_cols + self.item_cols + ["item_id"]
        self.specs = grammar()
        self.groups = {"static": self.cat_cols + ["release_year", "weekday"], "manual": [],
                       "recency": ["user_recency", "item_recency", "pair_count", "user_span", "user_unique_items"],
                       "trends": [f"{r}__trend_7_30" for r in CORE],
                       "multigenre": ["genre_overlap_last", "genre_affinity_mean", "genre_affinity_max"]}
        for s in self.specs:
            self.groups.setdefault(s.group, []).append(s.name)

    def enrich(self, rows):
        out = rows.copy()
        for col in self.user_cols:
            out[col] = out.user_id.map(self.users[col]).fillna("__MISSING__").astype(str)
        for col in self.item_cols:
            out[col] = out.item_id.map(self.items[col]).fillna("__MISSING__").astype(str)
        out["release_year"] = pd.to_numeric(out.item_id.map(self.items.release_year), errors="coerce").fillna(0).astype(np.float32)
        return out

    @property
    def baseline(self):
        return self.groups["static"] + self.groups["manual"] + self.groups["recency"]

    @property
    def automatic_groups(self):
        return [g for g in self.groups if g not in ["static", "manual", "recency"]]

    def registry(self):
        return {"grammar": [asdict(s) for s in self.specs], "groups": self.groups,
                "baseline": self.baseline, "categorical": self.cat_cols}

    @staticmethod
    def lookup(history, query, keys, weights=None):
        if history.empty:
            return np.zeros(len(query), dtype=np.float32)
        h = history[keys].copy()
        h["_v"] = 1.0 if weights is None else np.asarray(weights)
        agg = h.groupby(keys, sort=False, observed=True)._v.sum()
        ix = query[keys[0]] if len(keys) == 1 else pd.MultiIndex.from_frame(query[keys])
        return agg.reindex(ix).fillna(0).to_numpy(dtype=np.float32)

    def build(self, rows, history, day):
        day = pd.Timestamp(day)
        if len(history) and not (history.date < day).all():
            raise ValueError("History must be strictly earlier than prediction day")
        q = self.enrich(rows)
        h = self.enrich(history)
        q["weekday"] = np.float32(day.dayofweek)
        counts = {}
        for w in [7, 30, 0, 1, 3, 10, 14, 60]:
            hw = h if w == 0 else h[h.date >= day - pd.Timedelta(days=w)]
            for rel in CORE + (["segment_item"] if w in [7, 30, 0] else []):
                counts[(rel, w)] = self.lookup(hw, q, RELATIONS[rel])
        feature_values = {}
        for spec in self.specs:
            if spec.operation == "count":
                value = counts[(spec.relation, spec.window_days)]
            elif spec.operation == "share":
                # Denominator smoothing: count / (user_count + 1), not a CTR.
                value = counts[(spec.relation, spec.window_days)] / (counts[("user", spec.window_days)] + 1)
            else:
                weights = np.exp2(-(day - h.date).dt.days.to_numpy() / spec.half_life)
                value = self.lookup(h, q, RELATIONS[spec.relation], weights)
            feature_values[spec.name] = value
        q = pd.concat([q, pd.DataFrame(feature_values, index=q.index)], axis=1)
        for rel in CORE:
            q[f"{rel}__trend_7_30"] = (counts[(rel, 7)] / 7) / (counts[(rel, 30)] / 30 + .1)
        for rel in ["user", "item"]:
            key = RELATIONS[rel][0]
            last = h.groupby(key).date.max()
            q[f"{rel}_recency"] = (day - q[key].map(last)).dt.days.fillna(999).astype(np.float32)
        q["pair_count"] = self.lookup(h, q, ["user_id", "item_id"])
        first = h.groupby("user_id").date.min()
        q["user_span"] = (day - q.user_id.map(first)).dt.days.fillna(0).astype(np.float32)
        distinct = h.groupby("user_id").item_id.nunique()
        q["user_unique_items"] = q.user_id.map(distinct).fillna(0).astype(np.float32)
        # Genre affinity uses *all* listed genres, including secondary genres.
        recent = h[h.date >= day - pd.Timedelta(days=30)]
        profiles = {}
        for uid, part in recent.groupby("user_id", sort=False):
            from collections import Counter
            c = Counter(t for iid in part.item_id for t in self.genre_tokens.get(iid, ()))
            profiles[uid] = (c, len(part))
        last_items = h.sort_values(["date", "item_id"]).groupby("user_id").item_id.last().to_dict()
        multi = []
        for uid, iid in zip(q.user_id, q.item_id):
            ts = self.genre_tokens.get(iid, ())
            c, total = profiles.get(uid, ({}, 0))
            vals = [c.get(t, 0) / (total + 1) for t in ts] or [0.0]
            prev = set(self.genre_tokens.get(last_items.get(uid), ()))
            cur = set(ts)
            multi.append((len(prev & cur) / max(1, len(prev | cur)), np.mean(vals), max(vals)))
        multi = np.asarray(multi, dtype=np.float32).reshape(-1, 3)
        for i, col in enumerate(self.groups["multigenre"]):
            q[col] = multi[:, i]
        for col in set(sum(self.groups.values(), [])) - set(self.cat_cols):
            q[col] = q[col].astype(np.float32)
        return q
