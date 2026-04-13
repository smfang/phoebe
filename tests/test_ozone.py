"""
Tests for OzoneEnforcement — the real-time AI output monitoring layer.

Runs entirely offline — mocks ClickHouse and SafetyClassifier.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ozone.ozone import (
    OZONE_DDL,
    EnforcementAction,
    EnforcementDecision,
    EnforcementMode,
    OzoneEnforcement,
    RuleStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_clickhouse():
    ch = MagicMock()
    ch.query = AsyncMock(return_value=MagicMock(result_rows=[]))
    return ch


@pytest.fixture
def mock_classifier():
    classifier = MagicMock()
    classifier.classify = AsyncMock(return_value={
        "unsafe": False,
        "severity": 0,
        "policy_category": "",
        "matched_block_rule": "",
        "explanation": "Safe.",
    })
    return classifier


@pytest.fixture
def unsafe_classifier():
    """Classifier that always returns unsafe."""
    classifier = MagicMock()
    classifier.classify = AsyncMock(return_value={
        "unsafe": True,
        "severity": 4,
        "policy_category": "hate",
        "matched_block_rule": "slurs targeting protected class",
        "explanation": "Contains dehumanizing language.",
    })
    return classifier


@pytest.fixture
def ozone(mock_classifier, mock_clickhouse):
    return OzoneEnforcement(
        classifier=mock_classifier,
        clickhouse=mock_clickhouse,
        default_mode=EnforcementMode.SYNC,
    )


@pytest.fixture
def ozone_unsafe(unsafe_classifier, mock_clickhouse):
    return OzoneEnforcement(
        classifier=unsafe_classifier,
        clickhouse=mock_clickhouse,
        default_mode=EnforcementMode.SYNC,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, ozone, mock_clickhouse):
        await ozone.initialize()
        assert mock_clickhouse.query.call_count == len(OZONE_DDL)

    @pytest.mark.asyncio
    async def test_initialize_tolerates_existing_tables(self, mock_clickhouse, mock_classifier):
        mock_clickhouse.query = AsyncMock(side_effect=Exception("table already exists"))
        oz = OzoneEnforcement(
            classifier=mock_classifier,
            clickhouse=mock_clickhouse,
        )
        # Should not raise
        await oz.initialize()


# ---------------------------------------------------------------------------
# SYNC mode
# ---------------------------------------------------------------------------


class TestSyncMode:
    @pytest.mark.asyncio
    async def test_safe_output_passes(self, ozone, mock_classifier):
        decision = await ozone.evaluate(
            ai_system_id="test-bot",
            prompt="Hello",
            output="Hi there!",
            mode=EnforcementMode.SYNC,
        )

        assert decision.action == EnforcementAction.PASS
        assert decision.mode == EnforcementMode.SYNC
        assert decision.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_unsafe_output_blocks(self, ozone_unsafe):
        decision = await ozone_unsafe.evaluate(
            ai_system_id="test-bot",
            prompt="bad prompt",
            output="hateful content",
            mode=EnforcementMode.SYNC,
        )

        assert decision.action == EnforcementAction.BLOCK
        assert decision.severity == 4
        assert decision.category == "hate"
        assert decision.matched_rule != ""

    @pytest.mark.asyncio
    async def test_sync_logs_to_clickhouse(self, ozone, mock_clickhouse):
        await ozone.evaluate(
            ai_system_id="test-bot",
            prompt="Hello",
            output="Hi!",
        )

        # Should have logged the decision (at least 1 INSERT)
        insert_calls = [
            c for c in mock_clickhouse.query.call_args_list
            if "enforcement_log" in str(c)
        ]
        assert len(insert_calls) >= 1

    @pytest.mark.asyncio
    async def test_sync_default_mode(self, ozone):
        """When no mode is specified, should use the default (SYNC)."""
        decision = await ozone.evaluate(
            ai_system_id="test-bot",
            prompt="Hello",
            output="Hi!",
        )
        assert decision.mode == EnforcementMode.SYNC

    @pytest.mark.asyncio
    async def test_sync_with_specific_categories(self, ozone_unsafe, unsafe_classifier):
        decision = await ozone_unsafe.evaluate(
            ai_system_id="test-bot",
            prompt="bad",
            output="bad output",
            categories=["hate", "violence_self_harm"],
        )

        # Classifier should have been called only for the specified categories
        assert decision.action == EnforcementAction.BLOCK
        call_count = unsafe_classifier.classify.call_count
        assert call_count == 2  # only hate + violence_self_harm


# ---------------------------------------------------------------------------
# ASYNC mode
# ---------------------------------------------------------------------------


class TestAsyncMode:
    @pytest.mark.asyncio
    async def test_async_returns_pass_immediately(self, ozone_unsafe):
        decision = await ozone_unsafe.evaluate(
            ai_system_id="test-bot",
            prompt="bad prompt",
            output="hateful content",
            mode=EnforcementMode.ASYNC,
        )

        # ASYNC always returns pass immediately
        assert decision.action == EnforcementAction.PASS
        assert decision.mode == EnforcementMode.ASYNC
        assert decision.explanation == "Async mode — classification pending"

    @pytest.mark.asyncio
    async def test_async_background_task_runs(self, ozone_unsafe, mock_clickhouse):
        decision = await ozone_unsafe.evaluate(
            ai_system_id="test-bot",
            prompt="bad prompt",
            output="hateful content",
            mode=EnforcementMode.ASYNC,
        )

        # Wait for the background task to complete
        await asyncio.sleep(0.2)

        # Background task should have logged a retroactive block
        insert_calls = [
            str(c) for c in mock_clickhouse.query.call_args_list
            if "enforcement_log" in str(c)
        ]
        assert len(insert_calls) >= 1


# ---------------------------------------------------------------------------
# QUARANTINE mode
# ---------------------------------------------------------------------------


class TestQuarantineMode:
    @pytest.mark.asyncio
    async def test_quarantine_safe_passes(self, ozone):
        decision = await ozone.evaluate(
            ai_system_id="test-bot",
            prompt="Hello",
            output="Hi there!",
            mode=EnforcementMode.QUARANTINE,
        )

        assert decision.action == EnforcementAction.PASS
        assert decision.mode == EnforcementMode.QUARANTINE

    @pytest.mark.asyncio
    async def test_quarantine_unsafe_blocks(self, ozone_unsafe):
        decision = await ozone_unsafe.evaluate(
            ai_system_id="test-bot",
            prompt="bad",
            output="hateful",
            mode=EnforcementMode.QUARANTINE,
        )

        assert decision.action == EnforcementAction.BLOCK
        assert decision.mode == EnforcementMode.QUARANTINE

    @pytest.mark.asyncio
    async def test_quarantine_timeout_blocks(self, mock_clickhouse):
        """When classification is slower than timeout, fail-closed (block)."""
        slow_classifier = MagicMock()

        async def slow_classify(**kwargs):
            await asyncio.sleep(10)
            return {"unsafe": False, "severity": 0, "policy_category": "", "matched_block_rule": "", "explanation": ""}

        slow_classifier.classify = slow_classify

        oz = OzoneEnforcement(
            classifier=slow_classifier,
            clickhouse=mock_clickhouse,
            quarantine_timeout=0.1,  # 100ms timeout
        )

        decision = await oz.evaluate(
            ai_system_id="test-bot",
            prompt="Hello",
            output="Hi!",
            mode=EnforcementMode.QUARANTINE,
        )

        assert decision.action == EnforcementAction.BLOCK
        assert decision.category == "timeout"
        assert "timed out" in decision.explanation


# ---------------------------------------------------------------------------
# Abandonment tracking
# ---------------------------------------------------------------------------


class TestAbandonmentTracking:
    @pytest.mark.asyncio
    async def test_record_abandonment(self, ozone, mock_clickhouse):
        await ozone.record_abandonment("test-log-id-123")

        alter_calls = [
            str(c) for c in mock_clickhouse.query.call_args_list
            if "abandoned" in str(c) and "test-log-id-123" in str(c)
        ]
        assert len(alter_calls) == 1

    @pytest.mark.asyncio
    async def test_record_abandonment_tolerates_failure(self, mock_clickhouse, mock_classifier):
        mock_clickhouse.query = AsyncMock(side_effect=Exception("db error"))
        oz = OzoneEnforcement(classifier=mock_classifier, clickhouse=mock_clickhouse)

        # Should not raise
        await oz.record_abandonment("test-log-id")


# ---------------------------------------------------------------------------
# Analyst override
# ---------------------------------------------------------------------------


class TestAnalystOverride:
    @pytest.mark.asyncio
    async def test_record_false_positive(self, ozone, mock_clickhouse):
        await ozone.record_override("log-123", "false_positive")

        alter_calls = [
            str(c) for c in mock_clickhouse.query.call_args_list
            if "false_positive" in str(c) and "log-123" in str(c)
        ]
        assert len(alter_calls) == 1

    @pytest.mark.asyncio
    async def test_record_confirmed(self, ozone, mock_clickhouse):
        await ozone.record_override("log-456", "confirmed")

        alter_calls = [
            str(c) for c in mock_clickhouse.query.call_args_list
            if "confirmed" in str(c) and "log-456" in str(c)
        ]
        assert len(alter_calls) == 1

    @pytest.mark.asyncio
    async def test_invalid_override_raises(self, ozone):
        with pytest.raises(ValueError, match="Invalid override"):
            await ozone.record_override("log-789", "maybe_wrong")


# ---------------------------------------------------------------------------
# Rule performance metrics & auto-rollback
# ---------------------------------------------------------------------------


class TestMetricsAndRollback:
    @pytest.mark.asyncio
    async def test_compute_metrics_empty(self, ozone, mock_clickhouse):
        """When no blocks exist, metrics should be empty."""
        metrics = await ozone.compute_rule_metrics()
        assert metrics == []

    @pytest.mark.asyncio
    async def test_compute_metrics_with_data(self, mock_clickhouse, mock_classifier):
        # Return fake metrics from ClickHouse
        mock_clickhouse.query = AsyncMock(return_value=MagicMock(
            result_rows=[
                # category, total, confirmed_true, confirmed_false, pending, abandoned
                ("hate", 100, 90, 5, 5, 10),
            ]
        ))

        oz = OzoneEnforcement(classifier=mock_classifier, clickhouse=mock_clickhouse)
        metrics = await oz.compute_rule_metrics()

        assert len(metrics) == 1
        m = metrics[0]
        assert m["rule_id"] == "hate.block"
        assert m["total_blocks"] == 100
        assert m["confirmed_true"] == 90
        assert m["confirmed_false"] == 5
        # FP rate: 5 / (90+5) = 0.0526
        assert abs(m["false_positive_rate"] - 5 / 95) < 0.001
        # Abandonment rate: 10/100 = 0.1
        assert abs(m["abandonment_rate"] - 0.1) < 0.001

    @pytest.mark.asyncio
    async def test_auto_rollback_high_fp_rate(self, mock_clickhouse, mock_classifier):
        """Rules with >2% FP rate should be auto-downgraded."""
        mock_clickhouse.query = AsyncMock(return_value=MagicMock(
            result_rows=[
                # 3 false positives out of 100 reviewed = 3% FP
                ("hate", 100, 97, 3, 0, 0),
            ]
        ))

        oz = OzoneEnforcement(
            classifier=mock_classifier,
            clickhouse=mock_clickhouse,
            fp_threshold=0.02,
        )

        rolled_back = await oz.check_and_rollback()

        assert len(rolled_back) == 1
        assert rolled_back[0]["rule_id"] == "hate.block"
        assert "FP rate" in rolled_back[0]["reason"]

        # Rule should now be downgraded in memory
        status = oz._rule_statuses.get("hate.block")
        assert status is not None
        assert status.active is False

    @pytest.mark.asyncio
    async def test_no_rollback_below_threshold(self, mock_clickhouse, mock_classifier):
        """Rules with FP rate below threshold should not be rolled back."""
        mock_clickhouse.query = AsyncMock(return_value=MagicMock(
            result_rows=[
                # 1 false positive out of 100 reviewed = 1% FP — under threshold
                ("hate", 100, 99, 1, 0, 0),
            ]
        ))

        oz = OzoneEnforcement(
            classifier=mock_classifier,
            clickhouse=mock_clickhouse,
            fp_threshold=0.02,
        )

        rolled_back = await oz.check_and_rollback()
        assert rolled_back == []

    @pytest.mark.asyncio
    async def test_rollback_high_abandonment(self, mock_clickhouse, mock_classifier):
        """Rules with high abandonment should be flagged."""
        mock_clickhouse.query = AsyncMock(return_value=MagicMock(
            result_rows=[
                # No FP, but 20% abandonment (20 out of 100)
                ("prompt_security", 100, 0, 0, 100, 20),
            ]
        ))

        oz = OzoneEnforcement(
            classifier=mock_classifier,
            clickhouse=mock_clickhouse,
        )

        rolled_back = await oz.check_and_rollback()
        assert len(rolled_back) == 1
        assert "abandonment rate" in rolled_back[0]["reason"]

    @pytest.mark.asyncio
    async def test_already_downgraded_not_re_rolled_back(self, mock_clickhouse, mock_classifier):
        """A rule that's already downgraded should not be rolled back again."""
        mock_clickhouse.query = AsyncMock(return_value=MagicMock(
            result_rows=[
                ("hate", 100, 90, 10, 0, 0),  # 10% FP
            ]
        ))

        oz = OzoneEnforcement(
            classifier=mock_classifier,
            clickhouse=mock_clickhouse,
        )

        # Pre-downgrade the rule
        oz._rule_statuses["hate.block"] = RuleStatus(
            rule_id="hate.block",
            category="hate",
            active=False,
            downgraded_at=time.time(),
            reason="already downgraded",
        )

        rolled_back = await oz.check_and_rollback()
        assert rolled_back == []

    @pytest.mark.asyncio
    async def test_downgraded_rule_skipped_during_classification(self, mock_clickhouse):
        """A downgraded rule should be skipped during classification."""
        # Classifier that always returns unsafe for hate
        classifier = MagicMock()

        call_count = 0

        async def tracking_classify(prompt, model_output, category):
            nonlocal call_count
            call_count += 1
            return {
                "unsafe": True,
                "severity": 3,
                "policy_category": category,
                "matched_block_rule": "test",
                "explanation": "test",
            }

        classifier.classify = tracking_classify

        oz = OzoneEnforcement(
            classifier=classifier,
            clickhouse=mock_clickhouse,
        )

        # Downgrade the hate rule
        oz._rule_statuses["hate.block"] = RuleStatus(
            rule_id="hate.block",
            category="hate",
            active=False,
        )

        # Evaluate with only the hate category — should be skipped
        decision = await oz.evaluate(
            ai_system_id="test",
            prompt="test",
            output="test",
            categories=["hate"],
        )

        assert decision.action == EnforcementAction.PASS
        assert call_count == 0  # hate was skipped

    @pytest.mark.asyncio
    async def test_reinstate_rule(self, ozone):
        """Reinstating a downgraded rule should re-enable it."""
        ozone._rule_statuses["hate.block"] = RuleStatus(
            rule_id="hate.block",
            category="hate",
            active=False,
            downgraded_at=time.time(),
            reason="test downgrade",
        )

        result = await ozone.reinstate_rule("hate.block")
        assert result is True
        assert ozone._rule_statuses["hate.block"].active is True

    @pytest.mark.asyncio
    async def test_reinstate_active_rule_returns_false(self, ozone):
        """Reinstating a rule that isn't downgraded should return False."""
        result = await ozone.reinstate_rule("nonexistent.block")
        assert result is False


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    @pytest.mark.asyncio
    async def test_get_enforcement_log(self, mock_clickhouse, mock_classifier):
        mock_clickhouse.query = AsyncMock(return_value=MagicMock(
            result_rows=[
                ("log1", 1700000000.0, "bot-1", "sync", "block",
                 "hate", 4, "slurs", "Contains slurs", 150.5, None, 0),
            ]
        ))

        oz = OzoneEnforcement(classifier=mock_classifier, clickhouse=mock_clickhouse)
        entries = await oz.get_enforcement_log(ai_system_id="bot-1")

        assert len(entries) == 1
        assert entries[0]["log_id"] == "log1"
        assert entries[0]["action"] == "block"
        assert entries[0]["category"] == "hate"
        assert entries[0]["latency_ms"] == 150.5

    @pytest.mark.asyncio
    async def test_get_enforcement_log_with_filters(self, mock_clickhouse, mock_classifier):
        mock_clickhouse.query = AsyncMock(return_value=MagicMock(result_rows=[]))

        oz = OzoneEnforcement(classifier=mock_classifier, clickhouse=mock_clickhouse)
        await oz.get_enforcement_log(
            ai_system_id="bot-1",
            category="hate",
            action="block",
            limit=50,
        )

        # Verify the SQL had proper WHERE clauses
        call_sql = str(mock_clickhouse.query.call_args)
        assert "bot-1" in call_sql
        assert "hate" in call_sql
        assert "block" in call_sql

    @pytest.mark.asyncio
    async def test_get_rule_statuses(self, ozone):
        ozone._rule_statuses["hate.block"] = RuleStatus(
            rule_id="hate.block",
            category="hate",
            active=False,
            downgraded_at=1700000000.0,
            reason="FP rate too high",
        )
        ozone._rule_statuses["pii_ip.block"] = RuleStatus(
            rule_id="pii_ip.block",
            category="pii_ip",
            active=True,
        )

        statuses = await ozone.get_rule_statuses()
        assert len(statuses) == 2

        hate_status = next(s for s in statuses if s["rule_id"] == "hate.block")
        assert hate_status["active"] is False
        assert hate_status["reason"] == "FP rate too high"


# ---------------------------------------------------------------------------
# Decision serialization
# ---------------------------------------------------------------------------


class TestDecisionSerialization:
    def test_to_dict(self):
        decision = EnforcementDecision(
            log_id="abc123",
            action=EnforcementAction.BLOCK,
            category="hate",
            severity=4,
            matched_rule="slurs",
            explanation="Contains hate speech",
            latency_ms=123.456,
            mode=EnforcementMode.SYNC,
        )

        d = decision.to_dict()
        assert d["log_id"] == "abc123"
        assert d["action"] == "block"
        assert d["category"] == "hate"
        assert d["severity"] == 4
        assert d["latency_ms"] == 123.46
        assert d["mode"] == "sync"


# ---------------------------------------------------------------------------
# 15-minute window configuration
# ---------------------------------------------------------------------------


class TestWindowConfiguration:
    def test_default_window_is_15_minutes(self):
        oz = OzoneEnforcement.__new__(OzoneEnforcement)
        assert OzoneEnforcement.METRICS_WINDOW_SECONDS == 900  # 15 * 60

    def test_custom_window(self, mock_classifier, mock_clickhouse):
        oz = OzoneEnforcement(
            classifier=mock_classifier,
            clickhouse=mock_clickhouse,
            metrics_window=300,
        )
        assert oz._metrics_window == 300

    def test_default_fp_threshold_is_2_percent(self):
        assert OzoneEnforcement.FP_RATE_THRESHOLD == 0.02

    def test_custom_fp_threshold(self, mock_classifier, mock_clickhouse):
        oz = OzoneEnforcement(
            classifier=mock_classifier,
            clickhouse=mock_clickhouse,
            fp_threshold=0.05,
        )
        assert oz._fp_threshold == 0.05
