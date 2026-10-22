"""Recall tests: the first value must always be the current one, with a reason."""

import unittest

from pangu import Memory
from pangu.context import render, estimate_tokens


class TestRecall(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Memory(":memory:")
        self.m.write("我九月从深圳搬到北京")
        self.m.write("我住昌平")

    def tearDown(self) -> None:
        self.m.close()

    def test_returns_current_address_first(self):
        hits = self.m.recall("她现在住哪")
        self.assertTrue(hits)
        self.assertEqual(hits[0].predicate, "live_at")
        self.assertIn("昌平", hits[0].content)

    def test_every_hit_has_a_reason(self):
        hits = self.m.recall("她现在住哪")
        for h in hits:
            self.assertTrue(h.why, "a hit without a why must not be returned")
            self.assertIn("时效", h.why)

    def test_only_one_live_value_per_functional_predicate(self):
        self.m.write("我住上海")
        hits = [h for h in self.m.recall("她现在住哪", limit=20) if h.predicate == "live_at" and not h.stale]
        self.assertEqual(len(hits), 1, "two contradicting current addresses must never reach the model")
        self.assertIn("上海", hits[0].content)

    def test_history_query_surfaces_old_value(self):
        # "我九月从深圳搬到北京" derived live_at=北京; "我住昌平" superseded it.
        # Asking about the past must reach 北京 -- and mark it as no longer valid.
        hist = self.m.recall("她以前住哪")
        self.assertTrue(any("北京" in h.content for h in hist), "asking about the past must reach the past")
        beijing = next(h for h in hist if "北京" in h.content)
        self.assertTrue(beijing.stale, "a past address must be flagged stale")

    def test_present_query_hides_old_value(self):
        hist = self.m.recall("她现在住哪")
        self.assertFalse(any("深圳" in h.content for h in hist))

    def test_no_pollution_from_unrelated_writes(self):
        baseline = self.m.recall("她现在住哪")[0].content
        for i in range(200):
            self.m.write(f"我昨天看了一部叫测试{i}的电影")
        after = self.m.recall("她现在住哪")[0].content
        self.assertEqual(baseline, after, "200 unrelated memories must not move the answer")

    def test_context_block_is_short_and_budgeted(self):
        text = self.m.context_for("她现在住哪", max_tokens=120)
        self.assertLessEqual(estimate_tokens(text), 200)
        self.assertIn("昌平", text)

    def test_context_marks_stale_entries(self):
        hits = self.m.recall("她以前住哪")
        text = render(hits, include_why=True)
        self.assertIn("已失效", text)

    def test_empty_store_returns_nothing(self):
        fresh = Memory(":memory:")
        self.assertEqual(fresh.recall("她住哪"), [])
        self.assertEqual(fresh.context_for("她住哪"), "")
        fresh.close()


if __name__ == "__main__":
    unittest.main()
