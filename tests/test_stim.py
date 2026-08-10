"""Tests for fluxmem/stim.py -- STIM tier, LRU eviction."""

from __future__ import annotations

import pytest

from fluxmem.interfaces import Turn
from fluxmem.stim import ShortTermInteractionMemory


def _turn(turn_id: str, ts: float = 0.0) -> Turn:
    return Turn(turn_id=turn_id, user="u", assistant="a", timestamp=ts, last_access=ts)


class TestCapacityAndEviction:
    def test_fills_to_capacity_without_eviction(self) -> None:
        stim = ShortTermInteractionMemory(capacity=4)
        for i in range(4):
            evicted = stim.push(_turn(f"t{i}"))
            assert evicted is None
        assert len(stim) == 4

    def test_fifth_push_evicts_exactly_one(self) -> None:
        stim = ShortTermInteractionMemory(capacity=4)
        for i in range(4):
            stim.push(_turn(f"t{i}"))
        evicted = stim.push(_turn("t4"))
        assert evicted is not None
        assert evicted.turn_id == "t0"
        assert len(stim) == 4

    def test_touch_on_oldest_changes_which_turn_is_evicted(self) -> None:
        """The test that distinguishes LRU from FIFO."""
        stim = ShortTermInteractionMemory(capacity=4)
        for i in range(4):
            stim.push(_turn(f"t{i}"))
        # Without touch, t0 (oldest by insertion) would be evicted next.
        stim.touch("t0")
        evicted = stim.push(_turn("t4"))
        # t0 was just touched, so t1 -- now least-recently-used -- is evicted.
        assert evicted.turn_id == "t1"

    def test_eviction_order_under_interleaved_access(self) -> None:
        stim = ShortTermInteractionMemory(capacity=4)
        for i in range(4):
            stim.push(_turn(f"t{i}"))
        stim.touch("t1")
        stim.touch("t3")
        stim.touch("t0")
        # LRU order is now: t2 (never touched), t1, t3, t0.
        evicted = stim.push(_turn("t4"))
        assert evicted.turn_id == "t2"
        evicted = stim.push(_turn("t5"))
        assert evicted.turn_id == "t1"

    def test_capacity_one(self) -> None:
        stim = ShortTermInteractionMemory(capacity=1)
        assert stim.push(_turn("t0")) is None
        evicted = stim.push(_turn("t1"))
        assert evicted.turn_id == "t0"
        assert len(stim) == 1

    def test_capacity_zero_evicts_immediately(self) -> None:
        stim = ShortTermInteractionMemory(capacity=0)
        evicted = stim.push(_turn("t0"))
        assert evicted is not None
        assert evicted.turn_id == "t0"
        assert len(stim) == 0

    def test_negative_capacity_rejected(self) -> None:
        with pytest.raises(ValueError):
            ShortTermInteractionMemory(capacity=-1)


class TestTouchAndMembership:
    def test_touch_unknown_turn_raises(self) -> None:
        stim = ShortTermInteractionMemory(capacity=4)
        with pytest.raises(KeyError):
            stim.touch("missing")

    def test_contains(self) -> None:
        stim = ShortTermInteractionMemory(capacity=4)
        stim.push(_turn("t0"))
        assert "t0" in stim
        assert "missing" not in stim

    def test_peek_least_recently_used(self) -> None:
        stim = ShortTermInteractionMemory(capacity=4)
        assert stim.peek_least_recently_used() is None
        stim.push(_turn("t0"))
        stim.push(_turn("t1"))
        assert stim.peek_least_recently_used().turn_id == "t0"
        stim.touch("t0")
        assert stim.peek_least_recently_used().turn_id == "t1"

    def test_turns_ordered_lru_to_mru(self) -> None:
        stim = ShortTermInteractionMemory(capacity=4)
        for i in range(3):
            stim.push(_turn(f"t{i}"))
        stim.touch("t0")
        ids = [t.turn_id for t in stim.turns()]
        assert ids == ["t1", "t2", "t0"]

    def test_repushing_existing_turn_id_updates_recency(self) -> None:
        stim = ShortTermInteractionMemory(capacity=2)
        stim.push(_turn("t0"))
        stim.push(_turn("t1"))
        stim.push(_turn("t0", ts=99.0))
        assert len(stim) == 2
        assert stim.peek_least_recently_used().turn_id == "t1"
