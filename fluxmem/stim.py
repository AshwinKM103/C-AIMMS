"""fluxmem/stim.py -- STIM tier (COLM Sec 1.3.2).

Fixed-capacity buffer of the most recent raw dialogue turns.
"""

from __future__ import annotations

from collections import OrderedDict

from fluxmem.interfaces import Turn

DEFAULT_CAPACITY = 4


class ShortTermInteractionMemory:
    """Fixed-capacity buffer of raw dialogue turns, capacity 4 by default.

    Eviction is by LRU (last-access order), not insertion order (FIFO) --
    see FluxMem Eq. 5: `tau(p)` is defined as *"the last access time of page
    p,"* which is LRU, not the "oldest" reading of insertion order. `touch()`
    is the entire difference between the two policies: without it, the last
    access time of every turn equals its insertion time, and LRU degenerates
    to FIFO. Today nothing calls `touch()` (COLM Alg. 1's write phase never
    re-reads STIM), so this is currently FIFO in practice -- it starts to
    matter the moment a read path (ITERRET, out of scope here) touches STIM.

    Dependency-free by design: evicted turns are returned to the caller
    rather than pushed into MTEM directly, so STIM doesn't need to know MTEM
    exists (build-order requirement: stim.py precedes mtem.py).
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        self._capacity = capacity
        self._turns: OrderedDict[str, Turn] = OrderedDict()

    def __len__(self) -> int:
        return len(self._turns)

    def __contains__(self, turn_id: str) -> bool:
        return turn_id in self._turns

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, turn: Turn) -> Turn | None:
        """Inserts `turn` as most-recently-used; returns the evicted turn, if any.

        A capacity-0 buffer evicts every turn immediately (it is pushed and
        evicted in the same call).
        """
        if self._capacity == 0:
            return turn
        self._turns[turn.turn_id] = turn
        self._turns.move_to_end(turn.turn_id)
        if len(self._turns) <= self._capacity:
            return None
        _evicted_id, evicted_turn = self._turns.popitem(last=False)
        return evicted_turn

    def touch(self, turn_id: str) -> None:
        """Marks `turn_id` as most-recently-used without changing its content.

        This is the LRU update: the next eviction skips this turn in favor
        of whatever is now least-recently-used.
        """
        if turn_id not in self._turns:
            raise KeyError(turn_id)
        self._turns.move_to_end(turn_id)

    def peek_least_recently_used(self) -> Turn | None:
        """Returns the turn that would be evicted next, without evicting it."""
        if not self._turns:
            return None
        return next(iter(self._turns.values()))

    def turns(self) -> list[Turn]:
        """Returns buffered turns ordered least- to most-recently-used."""
        return list(self._turns.values())
