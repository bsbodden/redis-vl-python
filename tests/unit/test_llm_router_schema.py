"""Unit tests for LLM Router schema."""

import pytest
from pydantic import ValidationError

from redisvl.extensions.llm_router import ModelTier
from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    PretrainedReference,
    PretrainedRoute,
    PretrainedRouterConfig,
    RouteMatch,
    RoutingConfig,
)


class TestModelTier:
    """Tests for ModelTier schema."""

    def test_valid_tier(self):
        """Should create valid tier."""
        tier = ModelTier(
            name="simple",
            model="openai/gpt-4.1-nano",
            references=["hello", "hi"],
            distance_threshold=0.5,
        )
        assert tier.name == "simple"
        assert tier.model == "openai/gpt-4.1-nano"
        assert tier.references == ["hello", "hi"]
        assert tier.distance_threshold == 0.5

    def test_tier_with_metadata(self):
        """Should accept metadata."""
        tier = ModelTier(
            name="simple",
            model="test/model",
            references=["hello"],
            metadata={
                "cost_per_1k_input": 0.0001,
                "capabilities": ["chat"],
            },
        )
        assert tier.metadata["cost_per_1k_input"] == 0.0001
        assert "chat" in tier.metadata["capabilities"]

    def test_empty_name_fails(self):
        """Should reject empty name."""
        with pytest.raises(ValidationError):
            ModelTier(
                name="",
                model="test/model",
                references=["hello"],
            )

    def test_empty_model_fails(self):
        """Should reject empty model."""
        with pytest.raises(ValidationError):
            ModelTier(
                name="test",
                model="",
                references=["hello"],
            )

    def test_missing_model_fails(self):
        """Should reject missing model (required on ModelTier)."""
        with pytest.raises(ValidationError):
            ModelTier(
                name="test",
                references=["hello"],
            )

    def test_empty_references_fails(self):
        """Should reject empty references."""
        with pytest.raises(ValidationError):
            ModelTier(
                name="test",
                model="test/model",
                references=[],
            )

    def test_whitespace_reference_fails(self):
        """Should reject whitespace-only references."""
        with pytest.raises(ValidationError):
            ModelTier(
                name="test",
                model="test/model",
                references=["hello", "  "],
            )

    def test_threshold_bounds(self):
        """Should validate threshold bounds (0, 2]."""
        # Valid thresholds
        ModelTier(name="t", model="m", references=["r"], distance_threshold=0.1)
        ModelTier(name="t", model="m", references=["r"], distance_threshold=2.0)

        # Invalid: <= 0
        with pytest.raises(ValidationError):
            ModelTier(name="t", model="m", references=["r"], distance_threshold=0)

        with pytest.raises(ValidationError):
            ModelTier(name="t", model="m", references=["r"], distance_threshold=-0.1)

        # Invalid: > 2
        with pytest.raises(ValidationError):
            ModelTier(name="t", model="m", references=["r"], distance_threshold=2.1)


class TestRouteMatch:
    """Tests for RouteMatch schema."""

    def test_empty_match(self):
        """Empty match should be falsy."""
        match = RouteMatch()
        assert not match
        assert match.name is None
        assert match.model is None

    def test_valid_match(self):
        """Valid match should be truthy."""
        match = RouteMatch(
            name="simple",
            model="test/model",
            distance=0.3,
            confidence=0.85,
        )
        assert match
        assert match.name == "simple"
        assert match.confidence == 0.85

    def test_match_with_alternatives(self):
        """Should store alternative matches."""
        match = RouteMatch(
            name="simple",
            model="test/model",
            distance=0.3,
            alternatives=[("reasoning", 0.5), ("expert", 0.7)],
        )
        assert len(match.alternatives) == 2
        assert match.alternatives[0] == ("reasoning", 0.5)

    def test_match_with_metadata(self):
        """Should store route metadata."""
        match = RouteMatch(
            name="simple",
            model="test/model",
            metadata={"cost_per_1k_input": 0.0001},
        )
        assert match.metadata["cost_per_1k_input"] == 0.0001

    def test_tier_alias(self):
        """RouteMatch.tier should alias name for backward compat."""
        match = RouteMatch(name="simple")
        assert match.tier == "simple"


class TestRoutingConfig:
    """Tests for RoutingConfig schema."""

    def test_defaults(self):
        """Should have sensible defaults."""
        config = RoutingConfig()
        assert config.max_k == 1
        assert config.aggregation_method == DistanceAggregationMethod.avg
        assert config.cost_optimization is False
        assert config.cost_weight == 0.1
        assert config.default_route is None

    def test_custom_config(self):
        """Should accept custom values."""
        config = RoutingConfig(
            max_k=3,
            aggregation_method=DistanceAggregationMethod.min,
            cost_optimization=True,
            cost_weight=0.5,
            default_route="simple",
        )
        assert config.max_k == 3
        assert config.aggregation_method == DistanceAggregationMethod.min
        assert config.cost_optimization is True
        assert config.default_route == "simple"

    def test_cost_weight_bounds(self):
        """Cost weight should be 0-1."""
        RoutingConfig(cost_weight=0)
        RoutingConfig(cost_weight=1)

        with pytest.raises(ValidationError):
            RoutingConfig(cost_weight=-0.1)

        with pytest.raises(ValidationError):
            RoutingConfig(cost_weight=1.1)

    def test_max_k_positive(self):
        """max_k should be positive."""
        with pytest.raises(ValidationError):
            RoutingConfig(max_k=0)

        with pytest.raises(ValidationError):
            RoutingConfig(max_k=-1)


class TestPretrainedSchemas:
    """Tests for pretrained configuration schemas."""

    def test_pretrained_reference(self):
        """Should store text and vector."""
        ref = PretrainedReference(
            text="hello",
            vector=[0.1, 0.2, 0.3],
        )
        assert ref.text == "hello"
        assert ref.vector == [0.1, 0.2, 0.3]

    def test_pretrained_route(self):
        """Should store route with embedded references."""
        route = PretrainedRoute(
            name="simple",
            model="test/model",
            references=[
                PretrainedReference(text="hello", vector=[0.1, 0.2]),
                PretrainedReference(text="hi", vector=[0.3, 0.4]),
            ],
            distance_threshold=0.5,
        )
        assert route.name == "simple"
        assert len(route.references) == 2
        assert route.references[0].text == "hello"

    def test_pretrained_router_config(self):
        """Should store complete pretrained config."""
        config = PretrainedRouterConfig(
            name="test-router",
            version="1.0.0",
            vectorizer={"type": "hf", "model": "test-model"},
            routes=[
                PretrainedRoute(
                    name="simple",
                    model="test/model",
                    references=[
                        PretrainedReference(text="hello", vector=[0.1]),
                    ],
                )
            ],
        )
        assert config.name == "test-router"
        assert config.version == "1.0.0"
        assert len(config.routes) == 1

    def test_pretrained_router_config_legacy_tiers(self):
        """Should accept 'tiers' key for backward compatibility."""
        config = PretrainedRouterConfig(
            name="test-router",
            vectorizer={"type": "hf", "model": "test-model"},
            tiers=[
                PretrainedRoute(
                    name="simple",
                    model="test/model",
                    references=[
                        PretrainedReference(text="hello", vector=[0.1]),
                    ],
                )
            ],
        )
        assert len(config.routes) == 1


class TestDistanceAggregationMethod:
    """Tests for aggregation method enum."""

    def test_values(self):
        """Should have expected values."""
        assert DistanceAggregationMethod.avg.value == "avg"
        assert DistanceAggregationMethod.min.value == "min"
        assert DistanceAggregationMethod.sum.value == "sum"

    def test_from_string(self):
        """Should parse from string."""
        assert DistanceAggregationMethod("avg") == DistanceAggregationMethod.avg
        assert DistanceAggregationMethod("min") == DistanceAggregationMethod.min
