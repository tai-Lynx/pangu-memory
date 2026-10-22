"""Conflict resolution tests -- these are the moat, so they are the spec.

If these pass, claim #1 ("conflict resolution works") is defensible.
If any of them is removed, the project is a vector store with extra steps.
"""

import unittest
from datetime import datetime, timedelta, timezone

from pangu import Memory
from pangu.maintain import apply, decay_factor, forget
from pangu.models import Fact


def iso(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")


class TestWriteAndSupersede(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Memory(":memory:")

    def tearDown(self) -> None:
        self.m.close()

    def test_single_write_stores_fact(self):
        r = self.m.write("我住昌平")
        self.assertEqual(r.stored, 1)
        rows = self.m.store.live_for("self", "live_at")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].object, "昌平")

    def test_new_value_supersedes_old(self):
        self.m.write("我住深圳")
        self.m.write("我住昌平")

        live = self.m.store.live_for("self", "live_at")
        self.assertEqual(len(live), 1, "only one address may be current")
        self.assertEqual(live[0].object, "昌平")

        history = self.m.history(predicate="live_at")
        self.assertEqual(len(history), 2, "the old address must not be deleted")
        old = next(r for r in history if r.object == "深圳")
        self.assertIsNotNone(old.t_valid_to, "superseded row must have a closed interval")
        self.assertEqual(old.superseded_by, live[0].id, "superseded_by must point at the replacement")

    def test_duplicate_is_reinforced_not_duplicated(self):
        self.m.write("我住昌平")
        r = self.m.write("我住昌平")
        self.assertEqual(r.resolutions[0].action, "duplicate")
        self.assertEqual(len(self.m.store.live_for("self", "live_at")), 1)

    def test_move_event_implies_new_address(self):
        self.m.write("我九月从深圳搬到北京")
        live = self.m.store.live_for("self", "live_at")
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].object, "北京")
        self.assertEqual(live[0].meta.get("derived_from"), "moved_to")

    def test_past_anchor_does_not_compete_with_present(self):
        self.m.write("我小时候住福田")
        self.m.write("我住昌平")

        live = self.m.store.live_for("self", "live_at")
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].object, "昌平")

        history = self.m.history(predicate="live_at")
        self.assertTrue(any(r.object == "福田" for r in history), "childhood address is history, not noise")

    def test_retraction_closes_without_storing_a_fact(self):
        self.m.write("我住昌平")
        before = self.m.store.count()
        r = self.m.write("我不再住昌平了")

        self.assertEqual(r.resolutions[0].action, "retract")
        self.assertEqual(len(self.m.store.live_for("self", "live_at")), 0)
        self.assertEqual(self.m.store.count(), before, "a retraction is not a memory")

    def test_multivalued_predicate_accumulates(self):
        self.m.write("我喜欢咖啡")
        self.m.write("我喜欢茶")
        self.assertEqual(len(self.m.store.live_for("self", "likes")), 2)

    def test_unrelated_subject_does_not_conflict(self):
        self.m.write("我住昌平")
        self.m.remember("live_at", "海淀", subject="我妈")
        self.assertEqual(len(self.m.store.live_for("self", "live_at")), 1)
        self.assertEqual(len(self.m.store.live_for("我妈", "live_at")), 1)


class TestQuarantine(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Memory(":memory:")

    def tearDown(self) -> None:
        self.m.close()

    def test_low_confidence_new_fact_is_quarantined(self):
        self.m.write("我住昌平")
        weak = Fact(
            content="我住上海",
            predicate="live_at",
            object="上海",
            confidence=0.45,  # 0.45 + 0.15 < 0.85
        )
        _, res = apply(self.m.store, weak)
        self.assertEqual(res.action, "conflict")
        self.assertEqual(len(self.m.conflicts()), 1)

        live = self.m.store.live_for("self", "live_at")
        self.assertEqual(live[0].object, "昌平", "a weak claim must not overwrite a strong one")

    def test_backdated_claim_is_quarantined(self):
        self.m.write("我住昌平")
        backdated = Fact(
            content="我住上海",
            predicate="live_at",
            object="上海",
            t_valid_from=iso(30),
            confidence=0.9,
        )
        _, res = apply(self.m.store, backdated)
        self.assertEqual(res.action, "conflict", "claims to be current but starts in the past")

    def test_conflict_can_be_resolved_in_favour_of_new(self):
        self.m.write("我住昌平")
        apply(self.m.store, Fact(content="我住上海", predicate="live_at", object="上海", confidence=0.45))
        cid = self.m.conflicts()[0]["id"]
        self.m.resolve_conflict(cid, keep="new")

        self.assertEqual(len(self.m.conflicts()), 0)
        live = self.m.store.live_for("self", "live_at")
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].object, "上海")

    def test_conflict_can_be_resolved_in_favour_of_old(self):
        self.m.write("我住昌平")
        apply(self.m.store, Fact(content="我住上海", predicate="live_at", object="上海", confidence=0.45))
        cid = self.m.conflicts()[0]["id"]
        self.m.resolve_conflict(cid, keep="old")

        live = self.m.store.live_for("self", "live_at")
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].object, "昌平")


class TestMaintenance(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Memory(":memory:")

    def tearDown(self) -> None:
        self.m.close()

    def test_durable_facts_do_not_decay(self):
        self.m.write("我住昌平")
        row = self.m.store.live_for("self", "live_at")[0]
        far_future = datetime.now(timezone.utc) + timedelta(days=3650)
        self.assertEqual(decay_factor(row, far_future), 1.0)

    def test_events_decay(self):
        self.m.write("我九月从深圳搬到北京")
        event = next(r for r in self.m.store.all_rows(live_only=True) if r.predicate == "moved_to")
        later = datetime.now(timezone.utc) + timedelta(days=180)
        self.assertLess(decay_factor(event, later), 0.5)

    def test_weak_mentions_expire_too(self):
        self.m.write("我昨天看了一部叫《沙丘》的电影")
        row = next(r for r in self.m.store.all_rows(live_only=True) if r.predicate == "mentioned")
        later = datetime.now(timezone.utc) + timedelta(days=180)
        self.assertLess(decay_factor(row, later), 0.5)

    def test_forget_by_predicate(self):
        self.m.write("我住昌平")
        self.m.write("我喜欢咖啡")
        n = forget(self.m.store, predicate="live_at")
        self.assertEqual(n, 1)
        self.assertEqual(len(self.m.store.live_for("self", "live_at")), 0)
        self.assertEqual(len(self.m.store.live_for("self", "likes")), 1, "forget must be narrow")

    def test_audit_trail_exists(self):
        self.m.write("我住昌平")
        self.m.recall("我住哪")
        tail = self.m.audit_tail(5)
        ops = {row["op"] for row in tail}
        self.assertIn("write", ops)
        self.assertIn("recall", ops)


if __name__ == "__main__":
    unittest.main()
