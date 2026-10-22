"""Recall latency benchmark.

Target: 10,000 memories, recall P95 < 300 ms on a laptop CPU.

    python bench/bench_recall.py            # default 10k
    python bench/bench_recall.py --n 30000

The point is not to win a benchmark. The point is to know the number, so that
"we need a vector index" becomes a measured decision instead of a reflex.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pangu import Memory  # noqa: E402

CITIES = ["北京", "上海", "深圳", "广州", "杭州", "成都", "武汉", "西安", "南京", "重庆"]
FOODS = ["咖啡", "茶", "辣", "甜食", "火锅", "烧烤", "日料", "面食", "芒果", "香菜"]
TOPICS = ["编译器", "分布式系统", "强化学习", "字体设计", "黑胶唱片", "登山", "摄影", "围棋"]


def seed(m: Memory, n: int) -> None:
    """Write n memories through the real write path (conflict policy included)."""
    conn = m.store._conn
    rows = []
    for i in range(n):
        bucket = i % 4
        if bucket == 0:
            rows.append(("fact", "preference", f"我偶尔会想起{random.choice(TOPICS)}", f"topic{i}", 0.45, 0.3))
        elif bucket == 1:
            rows.append(("event", "mentioned", f"我{i}岁那年去过{random.choice(CITIES)}", f"trip{i}", 0.5, 0.35))
        elif bucket == 2:
            rows.append(("preference", "likes", f"我喜欢{random.choice(FOODS)}{i}", f"food{i}", 0.6, 0.45))
        else:
            rows.append(("fact", "mentioned", f"备注{i}：{random.choice(TOPICS)}相关的零碎记录", f"note{i}", 0.5, 0.3))

    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    conn.executemany(
        """INSERT INTO memory (kind, subject, subject_key, predicate, object, object_key,
                               content, t_valid_from, t_valid_to, anchor, confidence, importance,
                               source, fingerprint, quarantined, last_confirmed, created_at, meta)
           VALUES (?,?,?,?,?,?,?,?,NULL,'now',?,?,'bench',?,0,?,?, '{}')""",
        [
            (kind, "我", "self", pred, obj, obj.lower(), content, now, conf, imp, f"fp_{i}", now, now)
            for i, (kind, pred, content, obj, conf, imp) in enumerate(rows)
        ],
    )
    conn.commit()

    # A handful of real facts on top, so the benchmark measures a realistic mix.
    for text in ("我住昌平", "我在北京上学", "我喜欢咖啡"):
        m.write(text, silent=True)


def run(m: Memory, queries: list[str], rounds: int = 200) -> dict:
    latencies: list[float] = []
    for i in range(rounds):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        m.recall(q, limit=8)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "rounds": rounds,
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        "max_ms": round(latencies[-1], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--db", default=":memory:")
    args = ap.parse_args()

    m = Memory(args.db)
    t0 = time.perf_counter()
    seed(m, args.n)
    seed_s = time.perf_counter() - t0

    queries = [
        "她现在住哪",
        "她喜欢吃什么",
        "她在哪里上学",
        "她以前住在哪里",
        "她喜欢咖啡吗",
    ]
    result = run(m, queries)
    result["memories"] = m.store.count()
    result["seed_seconds"] = round(seed_s, 2)
    result["db"] = m.store.stats()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    verdict = "PASS" if result["p95_ms"] < 300 else "FAIL"
    print(f"\n{verdict}: P95 {result['p95_ms']} ms over {result['memories']} memories (target < 300 ms)")


if __name__ == "__main__":
    main()
