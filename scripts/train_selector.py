#!/usr/bin/env python
"""Thin CLI: train fluxmem's format selector against StubEpisodeProducer data.

All logic lives in fluxmem/; this script only wires config -> data -> fit.
With no real generation/retrieval pipeline (ITERRET/WORKMEM are non-goals),
`--dry-run` is the only mode this script actually supports -- it exercises
the full path (config -> stub episodes -> features -> reward labels ->
selector fit) against synthetic data, which is what `configs/default.yaml`
and `fluxmem.interfaces.StubEpisodeProducer` are for.
"""

from __future__ import annotations

import argparse

from fluxmem.config import load_config
from fluxmem.interfaces import FakeEntityExtractor, StubEpisodeProducer
from fluxmem.selector import FormatSelector
from fluxmem.supervision import ExactMatchF1Judge, TokenBudgetMemUtil, label_episodes


class _FixedResponseRunner:
    """Deterministic FormatRunner for --dry-run: every format "responds" identically.

    Real generation (ITERRET/WORKMEM) is out of scope for this package; this
    keeps the CLI runnable end-to-end without a live model.
    """

    def respond(self, episode, fmt):
        return episode.turns[0].assistant if episode.turns else ""

    def reference(self, episode):
        return episode.turns[0].assistant if episode.turns else ""

    def retrieve(self, episode, fmt):
        return [t.assistant for t in episode.turns]

    def gold_evidence(self, episode):
        return [episode.turns[0].assistant] if episode.turns else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an AdaStoreConfig YAML file")
    parser.add_argument("--n-episodes", type=int, default=40, help="Synthetic episodes to generate")
    parser.add_argument("--seed", type=int, default=0, help="StubEpisodeProducer seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Required: this package has no real generation pipeline (see module docstring)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    producer = StubEpisodeProducer(seed=args.seed)
    episodes = producer.produce(args.n_episodes)

    runner = _FixedResponseRunner()
    judge = ExactMatchF1Judge()
    mem_util = TokenBudgetMemUtil()
    entity_extractor = FakeEntityExtractor()

    X, y = label_episodes(episodes, runner, judge, mem_util, config.reward, entity_extractor)
    split = max(1, int(0.8 * len(X)))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:] or X_train, y[split:] or y_train

    selector = FormatSelector(config.selector)
    history = selector.fit(X_train, y_train, X_val, y_val)

    print(f"episodes={len(episodes)} train={len(X_train)} val={len(X_val)}")
    print(f"final_train_loss={history.train_loss[-1]:.4f} stopped_epoch={history.stopped_epoch}")


if __name__ == "__main__":
    main()
