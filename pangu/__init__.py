"""pangu-memory -- a local, explainable, forgettable long-term memory for LLMs.

The point of this library is not storage. It is the four operations that turn a
pile of embedded text into something that behaves like memory:

    conflict   when new information contradicts old, someone has to win
    decay      not everything stays equally relevant forever
    forget     explicit forgetting, with an auditable trail
    explain    every recall says why it fired

    >>> from pangu import Memory
    >>> m = Memory()                                  # ~/.pangu/db.sqlite
    >>> m.write("我九月从深圳搬到北京")
    >>> m.write("我住昌平")
    >>> print(m.context_for("她现在住哪"))
"""

from .models import Fact, Hit, MemoryRow, Resolution
from .memory import Memory, WriteResult

__all__ = [
    "Memory",
    "WriteResult",
    "Fact",
    "Hit",
    "MemoryRow",
    "Resolution",
]

__version__ = "0.1.0"
