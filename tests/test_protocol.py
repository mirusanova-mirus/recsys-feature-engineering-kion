import unittest

import numpy as np
import pandas as pd

from recsys.candidates import make_candidates
from recsys.build import split_bounds
from recsys.features import FeatureBuilder
from recsys.metrics import evaluate, paired_bootstrap


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.users = pd.DataFrame({"user_id": [1, 2], "age": ["a", "b"], "sex": ["m", "f"],
                                   "income": ["x", "x"], "kids_flg": [0, 1]})
        self.items = pd.DataFrame({"item_id": [10, 11, 12], "genres": ["a, b", "b", "c"],
                                   "countries": ["x", "x", "y"], "directors": ["d", "e", "f"],
                                   "content_type": ["film"] * 3, "release_year": [2000] * 3,
                                   "for_kids": [0] * 3, "age_rating": [16] * 3})
        self.h = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 11, 12],
                               "date": pd.to_datetime(["2021-01-01", "2021-01-07", "2021-01-02"])})
        self.q = pd.DataFrame({"user_id": [1, 2], "item_id": [12, 10]})

    def test_window_boundary_and_current_day_guard(self):
        b = FeatureBuilder(self.users, self.items)
        f = b.build(self.q, self.h, "2021-01-08")
        self.assertEqual(f.iloc[0]["user__count__7d"], 2)
        self.assertEqual(f.iloc[0]["user__count__1d"], 1)
        self.assertEqual(f.iloc[0].user_recency, 1)
        with self.assertRaises(ValueError):
            b.build(self.q, self.h, "2021-01-07")

    def test_future_invariance_and_no_target_columns(self):
        b = FeatureBuilder(self.users, self.items)
        augmented = pd.concat([self.h, pd.DataFrame({"user_id": [1, 1], "item_id": [12, 12],
                                                    "date": pd.to_datetime(["2021-01-08", "2021-01-20"])})])
        a = b.build(self.q, self.h, "2021-01-08")
        z = b.build(self.q, augmented[augmented.date < "2021-01-08"], "2021-01-08")
        pd.testing.assert_frame_equal(a, z)
        self.assertFalse(set(b.baseline) & {"label", "total_dur", "watched_pct", "date", "user_id"})

    def test_multi_genre_secondary_token(self):
        b = FeatureBuilder(self.users, self.items)
        q = pd.DataFrame({"user_id": [1], "item_id": [11]})
        h = self.h.iloc[[0]]
        f = b.build(q, h, "2021-01-08")
        self.assertEqual(f.iloc[0]["user_genre__count__0d"], 0)
        self.assertGreater(f.iloc[0].genre_affinity_mean, 0)

    def test_candidate_invariance_to_future_and_uniqueness(self):
        e = pd.DataFrame({"user_id": [2] * 9 + [1], "item_id": list(range(10, 19)) + [10],
                          "date": pd.to_datetime(["2021-01-01"] * 9 + ["2021-01-08"])})
        a, _ = make_candidates(e, "2021-01-08", negatives=4)
        future = pd.DataFrame({"user_id": [1], "item_id": [18], "date": pd.to_datetime(["2021-01-09"])})
        b, _ = make_candidates(pd.concat([e, future]), "2021-01-08", negatives=4)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.item_id.nunique(), len(a))
        self.assertEqual(a.label.sum(), 1)

    def test_metrics_hand_computed_ties_and_bootstrap(self):
        f = pd.DataFrame({"qid": [0, 0, 0, 1, 1], "user_id": [1, 1, 1, 2, 2],
                          "item_id": [10, 11, 12, 10, 11], "label": [0, 1, 0, 1, 0], "cold_user": [0] * 5})
        scores = np.array([.9, .8, .1, .9, .1])
        m, q = evaluate(f, scores)
        self.assertAlmostEqual(m["ndcg@5"], (1 / np.log2(3) + 1) / 2)
        self.assertAlmostEqual(m["mrr"], .75)
        self.assertEqual(paired_bootstrap(q, q, reps=30)["ci95"], [0, 0])
        with self.assertRaises(ValueError):
            paired_bootstrap(q, q.iloc[:1], reps=30)
        a, _ = evaluate(f, np.full(len(f), .5))
        b, _ = evaluate(f.iloc[::-1], np.full(len(f), .5))
        self.assertEqual(a, b)

    def test_catalog_coverage_repeats_and_multiple_targets(self):
        e = pd.DataFrame({"user_id": [1] + [2] * 9 + [1, 1, 1, 3, 3],
                          "item_id": [10] + list(range(11, 20)) + [10, 11, 99, 12, 13],
                          "date": pd.to_datetime(["2021-01-01"] * 10 + ["2021-01-08"] * 5)})
        rows, stats = make_candidates(e, "2021-01-08", negatives=4)
        self.assertEqual(stats["unknown_item_targets"], 1)
        self.assertEqual(stats["repeated_targets"], 1)
        self.assertEqual(stats["eligible_targets"], 3)
        self.assertEqual(set(rows[rows.user_id == 1].query("label == 1").item_id), {11})
        self.assertNotIn(10, set(rows[rows.user_id == 1].item_id))
        self.assertNotIn(99, set(rows.item_id))
        self.assertEqual(rows[rows.user_id == 3].cold_user.unique().tolist(), [1])

    def test_multi_positive_idcg(self):
        frame = pd.DataFrame({"qid": [4] * 4, "user_id": [1] * 4, "item_id": [10, 11, 12, 13],
                              "label": [1, 0, 1, 0], "cold_user": [0] * 4})
        metric, _ = evaluate(frame, [.9, .8, .7, .6])
        expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
        self.assertAlmostEqual(metric["ndcg@5"], expected)
        self.assertEqual(metric["recall@5"], 1)

    def test_temporal_replication_bounds_do_not_use_latest_day(self):
        events = pd.DataFrame({"date": pd.to_datetime(["2021-03-13", "2021-08-22"])})
        config = {"train_days": 28, "search_days": 7, "confirmation_days": 7, "test_days": 7}
        first = split_bounds(events, config)
        second = split_bounds(events, {**config, "end_exclusive": "2021-08-02"})
        self.assertEqual(second["train"], (pd.Timestamp("2021-06-14"), pd.Timestamp("2021-07-12")))
        self.assertEqual(second["test"], (pd.Timestamp("2021-07-26"), pd.Timestamp("2021-08-02")))
        self.assertLessEqual(second["test"][1], first["test"][0])
        self.assertEqual(list(second.values())[-1][1], pd.Timestamp("2021-08-02"))
        with self.assertRaises(ValueError):
            split_bounds(events, {**config, "end_exclusive": "2021-08-24"})
        with self.assertRaises(ValueError):
            split_bounds(events, {**config, "train_days": 0})


if __name__ == "__main__":
    unittest.main()
