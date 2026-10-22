"""End-to-end demo of the thing that makes this library different.

    python examples/quickstart.py

Everything here is made up. No real person's data appears in this file, and it
should stay that way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pangu import Memory  # noqa: E402

BAR = "─" * 62


def show(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def show_write(m: Memory, text: str) -> None:
    result = m.write(text)
    print(f"  写: {text}")
    for line in result.report().splitlines():
        print(f"    {line.strip()}")


def dump(m: Memory, query: str) -> None:
    print(f"\n  Q: {query}")
    hits = m.recall(query)
    if not hits:
        print("  （没有记忆）")
        return
    for h in hits:
        flag = "  [已失效]" if h.stale else ""
        print(f"  → {h.content}{flag}   score={h.score:.2f}")
        print(f"      why: {h.why}")


def main() -> None:
    m = Memory(":memory:")

    show("1. 两句互相矛盾的话，谁赢？")
    show_write(m, "我九月从深圳搬到北京")
    show_write(m, "我住昌平")
    print("\n  「搬到北京」派生出一条 live_at=北京，随即被「我住昌平」取代。")

    dump(m, "她现在住哪")
    dump(m, "她以前住哪")

    show("2. 历史不是矛盾")
    show_write(m, "我小时候住福田")
    dump(m, "她小时候住哪")
    print("\n  福田被存成闭区间：它是历史，永远不会跟「现在住哪」抢。")

    show("3. 多值谓词：喜欢的东西可以很多")
    show_write(m, "我喜欢咖啡")
    show_write(m, "我喜欢茶")
    dump(m, "她喜欢什么")

    show("4. 冲突隔离：机器不确定时不猜")
    rid, res = m.remember("live_at", "上海", confidence=0.45)
    print(f"  写入低置信度事实（conf=0.45）：{res}")
    print(f"  待裁决冲突: {len(m.conflicts())} 条")
    dump(m, "她现在住哪")
    print("\n  可疑事实已入库（可审计），但被隔离，不参与召回。")

    show("5. 显式遗忘")
    show_write(m, "我不再住昌平了")
    dump(m, "她现在住哪")
    print("\n  撤销句本身不入库 —— 它陈述「什么不成立」，不该被当作事实召回。")

    show("6. 撤销之后重新建立")
    show_write(m, "我住海淀")
    dump(m, "她现在住哪")

    show("7. 拼进 prompt 的那一段（带预算）")
    print(m.context_for("她现在住哪", max_tokens=200, include_why=True))

    show("8. 健康检查")
    for k, v in m.stats().items():
        print(f"  {k:16} {v}")
    print("\n  最近 4 条审计:")
    for row in m.audit_tail(4):
        print(f"  [{row['op']:6}] {row['query']}")

    m.close()
    print(f"\n{BAR}\n一个 sqlite 文件就是全部记忆。cp 走即备份。\n{BAR}")


if __name__ == "__main__":
    main()
