"""Context assembly: the few hundred tokens that actually reach the model.

Recalled memories are useless if they arrive as a JSON blob. This module
renders them as short, dated lines the model can read directly, under a hard
budget -- because the one thing a memory layer must never do is eat the
context window it was supposed to save.
"""

from __future__ import annotations

from .models import Hit, human_ts

DEFAULT_HEADER = "关于用户的长期记忆（按可信度排序）："
DEFAULT_FOOTER = "以上记忆若与用户当前说法冲突，以用户当前说法为准。"

#: Rough characters-per-token for mixed Chinese/English. Deliberately
#: pessimistic (real ratio is closer to 1.5) so we under-fill, not overflow.
CHARS_PER_TOKEN = 1.6


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def render(
    hits: list[Hit],
    *,
    max_tokens: int = 600,
    header: str = DEFAULT_HEADER,
    footer: str = DEFAULT_FOOTER,
    include_why: bool = False,
) -> str:
    """Render hits into a prompt-ready block that fits ``max_tokens``."""
    if not hits:
        return ""

    lines: list[str] = [header]
    used = estimate_tokens(header) + estimate_tokens(footer)

    for hit in hits:
        line = f"- {hit.content}"
        if hit.kind == "event":
            line += f"（{human_ts(hit.t_valid_from)}）"
        if hit.stale:
            line += "（已失效）"
        if include_why:
            line += f"  ← {hit.why}"
        cost = estimate_tokens(line)
        if used + cost > max_tokens:
            break
        lines.append(line)
        used += cost

    if len(lines) == 1:
        return ""
    lines.append(footer)
    return "\n".join(lines)


def context_for(
    store_or_hits,
    query: str | None = None,
    *,
    max_tokens: int = 600,
    include_why: bool = False,
    **recall_kwargs,
) -> str:
    """Convenience wrapper: recall then render."""
    from .recall import recall as _recall
    from .store.sqlite import SqliteStore

    if isinstance(store_or_hits, SqliteStore):
        if query is None:
            raise ValueError("query is required when passing a store")
        hits = _recall(store_or_hits, query, **recall_kwargs)
    else:
        hits = store_or_hits
    return render(hits, max_tokens=max_tokens, include_why=include_why)
