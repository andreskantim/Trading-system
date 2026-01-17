"""
Orchestration module for parallel execution of backtests.

This module provides different execution backends for running backtests
on a 16-core local workstation.
"""

from orchestration.orchestrator_factory import get_orchestrator

__all__ = ['get_orchestrator']
