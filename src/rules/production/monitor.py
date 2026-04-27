"""Performance monitoring and health checks.

Tracks system metrics and provides health status information.
"""

import time
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """System performance tracking."""

    prediction_count: int = 0
    error_count: int = 0
    total_processing_time: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate [0-1]."""
        if self.prediction_count == 0:
            return 0.0
        return (self.prediction_count - self.error_count) / self.prediction_count

    @property
    def avg_processing_time_ms(self) -> float:
        """Calculate average processing time in milliseconds."""
        if self.prediction_count == 0:
            return 0.0
        return self.total_processing_time / self.prediction_count


class Monitor:
    """System monitoring and health checks."""

    def __init__(self):
        """Initialize monitor."""
        self.metrics = PerformanceMetrics()
        self.bias_alerts_count = 0

    def record_prediction(
        self, processing_time_ms: float, success: bool, has_bias_alert: bool = False
    ):
        """Record prediction metrics.

        Args:
            processing_time_ms: Processing time in milliseconds
            success: Whether prediction succeeded
            has_bias_alert: Whether bias alert was triggered
        """
        self.metrics.prediction_count += 1
        self.metrics.total_processing_time += processing_time_ms

        if not success:
            self.metrics.error_count += 1

        if has_bias_alert:
            self.bias_alerts_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Dict with prediction counts, rates, and performance metrics
        """
        return {
            "prediction_count": self.metrics.prediction_count,
            "error_count": self.metrics.error_count,
            "success_rate": self.metrics.success_rate,
            "avg_processing_time_ms": self.metrics.avg_processing_time_ms,
            "bias_alerts_count": self.bias_alerts_count,
            "uptime_seconds": time.time() - self.metrics.start_time,
        }

    def health_check(self) -> Dict[str, Any]:
        """System health status.

        Returns:
            Dict with health status information
        """
        return {
            "status": "healthy" if self.metrics.success_rate > 0.95 else "degraded",
            "uptime_seconds": time.time() - self.metrics.start_time,
            "error_rate": (
                self.metrics.error_count / max(1, self.metrics.prediction_count)
            ),
            "prediction_count": self.metrics.prediction_count,
        }
