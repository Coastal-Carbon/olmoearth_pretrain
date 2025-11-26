"""
High precision console logger that shows more significant figures for gradient norms and other metrics.
"""

import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Dict, List, Optional

from olmo_core.train.callbacks.callback import Callback

log = logging.getLogger(__name__)


def format_float_high_precision(value: float) -> str:
    """Format float with higher precision than the default olmo_core format_float."""
    if value == 0.0:
        return "0.0"
    elif abs(value) < 1e-8:
        return f"{value:.2E}"
    elif abs(value) < 1e-6:
        return f"{value:.8f}"  # Show 8 decimal places for very small values
    elif abs(value) < 1e-3:
        return f"{value:.6f}"  # Show 6 decimal places for small values  
    elif abs(value) < 1:
        return f"{value:.5f}"  # Show 5 decimal places for values < 1
    elif abs(value) < 10:
        return f"{value:.4f}"  # Show 4 decimal places for single digits
    elif abs(value) < 100:
        return f"{value:.3f}"  # Show 3 decimal places for double digits
    elif abs(value) > 1000:
        return f"{int(value):,d}"
    else:
        return f"{value:.2f}"  # Show 2 decimal places for hundreds


@dataclass
class HighPrecisionConsoleLoggerCallback(Callback):
    """
    Console logger that shows more significant figures for metrics like gradient norms.
    
    This is particularly useful for debugging training issues where small values
    like gradient norms get rounded to 0.0 in the standard console output.
    """

    log_interval: int = 1
    """How often, in steps, to log progress to the console."""

    metrics_log_interval: Optional[int] = None
    """How often, in steps, to log metrics to the console. If not set, defaults to log_interval."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "train/*",
            "system/*", 
            "optim/*",  # Include all optim metrics with high precision
            "throughput/*",
        ]
    )
    """Metrics to log to the console with high precision. Wildcards are supported."""

    def post_step(self):
        if self._should_log_metrics(self.step):
            # Will log to console from `self.log_metrics()`.
            return

        if self.step % self.log_interval != 0:
            return

        log.info(self._get_progress_marker(self.step))

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if not self._should_log_metrics(step):
            return

        prefix = self._get_progress_marker(step, include_eta=True)
        
        # Filter and format metrics with high precision
        filtered_metrics = [
            f"    {name}={format_float_high_precision(value)}"
            for name, value in metrics.items()
            if any(fnmatch(name, pat) for pat in self.metrics)
        ]
        
        if filtered_metrics:
            log.info(f"{prefix}\n" + "\n".join(filtered_metrics))

    def _get_progress_marker(self, step: int, include_eta: bool = False) -> str:
        if include_eta and (eta := self.trainer.training_progress.time_remaining) is not None:
            from olmo_core.utils import format_timedelta
            eta_str = format_timedelta(eta).replace(", ", "")
            if self.trainer.hard_stop:
                eta_str = f"{eta_str}(hard stop)"
            return (
                f"[step={step}/{self.trainer.max_steps},epoch={self.trainer.epoch},eta={eta_str}]"
            )
        else:
            return f"[step={step}/{self.trainer.max_steps},epoch={self.trainer.epoch}]"

    def _should_log_metrics(self, step: int) -> bool:
        metrics_log_interval = self.metrics_log_interval or self.log_interval
        if step == 1 or (step > 1 and step % metrics_log_interval == 0):
            return True
        else:
            return False