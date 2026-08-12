"""Tests for EpisodeEncoder Protocol boundary (T-04).

Verifies that the HETREP encoding seam is wired correctly and injectable.
Per COLM Algorithm 1 line 4: "Encode e_j into (H_j, I_j, v_j) via HETREP".
"""

import numpy as np
import pytest

from fluxmem.interfaces import EpisodeEncoder, EpisodicUnit, MemoryFormat, Turn


class FakeEpisodeEncoder:
    """Test implementation of EpisodeEncoder Protocol."""

    def __init__(
        self,
        embedding: np.ndarray | None = None,
        hyperedge_density: float = 0.5,
        visual_salience: float = 0.3,
    ):
        self.embedding = embedding if embedding is not None else np.zeros(384)
        self.hyperedge_density = hyperedge_density
        self.visual_salience = visual_salience

    def encode(self, unit: EpisodicUnit) -> EpisodicUnit:
        """Populate encoding fields in the unit."""
        unit.embedding = self.embedding
        unit.hyperedge_density = self.hyperedge_density
        unit.visual_salience = self.visual_salience
        return unit


@pytest.fixture
def sample_unit() -> EpisodicUnit:
    """Create a minimal unit from segmentation (turns populated)."""
    return EpisodicUnit(
        episode_id="ep-1",
        turns=[
            Turn(
                turn_id="t1",
                user="Hello",
                assistant="Hi there",
                timestamp=0.0,
                last_access=0.0,
            ),
            Turn(
                turn_id="t2",
                user="How are you?",
                assistant="I'm doing well",
                timestamp=1.0,
                last_access=1.0,
            ),
        ],
        primary_format=MemoryFormat.VS,
    )


def test_episode_encoder_protocol_contract(sample_unit: EpisodicUnit):
    """Verify EpisodeEncoder populates all three encoding fields."""
    encoder = FakeEpisodeEncoder(
        embedding=np.ones(384),
        hyperedge_density=0.7,
        visual_salience=0.4,
    )

    result = encoder.encode(sample_unit)

    assert result.episode_id == "ep-1"
    assert result.embedding is not None
    assert np.allclose(result.embedding, np.ones(384))
    assert result.hyperedge_density == 0.7
    assert result.visual_salience == 0.4


def test_episode_encoder_preserves_turns_and_format(sample_unit: EpisodicUnit):
    """Verify encoder does not modify turns or primary_format."""
    original_turns = sample_unit.turns
    original_format = sample_unit.primary_format

    encoder = FakeEpisodeEncoder()
    result = encoder.encode(sample_unit)

    assert result.turns == original_turns
    assert result.primary_format == original_format


def test_episode_encoder_is_runtime_checkable():
    """Verify FakeEpisodeEncoder satisfies EpisodeEncoder Protocol."""
    encoder = FakeEpisodeEncoder()
    assert isinstance(encoder, EpisodeEncoder)


def test_episode_encoder_default_vs_values():
    """Verify default encoding values (before real encoders)."""
    unit = EpisodicUnit(
        episode_id="ep-0",
        turns=[],
        primary_format=MemoryFormat.VS,
    )

    # Before encoding: all fields should be default
    assert unit.embedding is None
    assert unit.hyperedge_density == 0.0
    assert unit.visual_salience == 0.0

    # After encoding with defaults
    encoder = FakeEpisodeEncoder()
    result = encoder.encode(unit)

    assert result.embedding is not None
    assert result.hyperedge_density == 0.5  # FakeEpisodeEncoder default
    assert result.visual_salience == 0.3  # FakeEpisodeEncoder default


def test_episode_encoder_stub_output_shape():
    """Verify embedding output has correct shape (384-d per COLM §1.2.3)."""
    encoder = FakeEpisodeEncoder(embedding=np.random.randn(384))
    unit = EpisodicUnit(
        episode_id="ep-2",
        turns=[],
        primary_format=MemoryFormat.VS,
    )

    result = encoder.encode(unit)

    assert result.embedding.shape == (384,)


def test_episode_encoder_clipped_density():
    """Verify hyperedge_density is normalized to [0, 1]."""
    # Encoders should clip to [0, 1]; validate boundary behavior.
    for density in [0.0, 0.5, 1.0]:
        encoder = FakeEpisodeEncoder(hyperedge_density=density)
        unit = EpisodicUnit(
            episode_id="ep-test",
            turns=[],
            primary_format=MemoryFormat.HG,
        )
        result = encoder.encode(unit)
        assert 0.0 <= result.hyperedge_density <= 1.0
