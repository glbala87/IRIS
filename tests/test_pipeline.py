"""Tests for IRIS pipeline orchestrator."""
import json
from pathlib import Path

import pytest


class TestCheckpoint:
    """Test checkpoint system."""

    def test_create_checkpoint(self, tmp_path):
        from iris.pipeline import Checkpoint
        cp = Checkpoint(tmp_path)
        assert cp.state['completed'] == []
        assert cp.state['failed'] == []

    def test_mark_done(self, tmp_path):
        from iris.pipeline import Checkpoint
        cp = Checkpoint(tmp_path)
        cp.mark_done('clustering', duration_s=5.0)
        assert cp.is_done('clustering')
        assert not cp.is_done('dtu')
        # Persists to disk
        cp2 = Checkpoint(tmp_path)
        assert cp2.is_done('clustering')

    def test_mark_failed(self, tmp_path):
        from iris.pipeline import Checkpoint
        cp = Checkpoint(tmp_path)
        cp.mark_failed('dtu', 'ValueError: test error')
        assert 'dtu' in cp.state['failed']
        assert 'test error' in cp.state['dtu_error']

    def test_mark_skipped(self, tmp_path):
        from iris.pipeline import Checkpoint
        cp = Checkpoint(tmp_path)
        cp.mark_skipped('ase', 'no BAM provided')
        assert 'ase' in cp.state['skipped']

    def test_reset(self, tmp_path):
        from iris.pipeline import Checkpoint
        cp = Checkpoint(tmp_path)
        cp.mark_done('clustering')
        cp.mark_failed('dtu', 'error')
        cp.reset()
        assert cp.state['completed'] == []
        assert cp.state['failed'] == []

    def test_resume_from_disk(self, tmp_path):
        from iris.pipeline import Checkpoint
        cp1 = Checkpoint(tmp_path)
        cp1.mark_done('clustering', 3.0)
        cp1.mark_done('dual_clustering', 8.0)

        # New instance loads from disk
        cp2 = Checkpoint(tmp_path)
        assert cp2.is_done('clustering')
        assert cp2.is_done('dual_clustering')
        assert not cp2.is_done('dtu')


class TestRunStep:
    """Test _run_step error handling."""

    def test_success(self, tmp_path):
        from iris.pipeline import _run_step, Checkpoint
        from iris._logging import get_main_logger
        logger = get_main_logger("test")
        cp = Checkpoint(tmp_path)

        def good_func():
            pass

        result = _run_step('test_step', good_func, cp, logger)
        assert result is True
        assert cp.is_done('test_step')

    def test_failure_recovers(self, tmp_path):
        from iris.pipeline import _run_step, Checkpoint
        from iris._logging import get_main_logger
        logger = get_main_logger("test")
        cp = Checkpoint(tmp_path)

        def bad_func():
            raise ValueError("intentional test error")

        result = _run_step('bad_step', bad_func, cp, logger)
        assert result is False
        assert 'bad_step' in cp.state['failed']
        # Doesn't crash the caller

    def test_skip_if_done(self, tmp_path):
        from iris.pipeline import _run_step, Checkpoint
        from iris._logging import get_main_logger
        logger = get_main_logger("test")
        cp = Checkpoint(tmp_path)
        cp.mark_done('done_step')

        call_count = [0]

        def should_not_run():
            call_count[0] += 1

        result = _run_step('done_step', should_not_run, cp, logger)
        assert result is True
        assert call_count[0] == 0  # wasn't called

    def test_force_reruns(self, tmp_path):
        from iris.pipeline import _run_step, Checkpoint
        from iris._logging import get_main_logger
        logger = get_main_logger("test")
        cp = Checkpoint(tmp_path)
        cp.mark_done('force_step')

        call_count = [0]

        def should_run():
            call_count[0] += 1

        result = _run_step(
            'force_step', should_run, cp, logger, force=True)
        assert result is True
        assert call_count[0] == 1  # was called despite being "done"
