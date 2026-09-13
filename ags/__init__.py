"""
AGS-DD: Adaptive Guidance Scheduling for Training-Free Diffusion Dataset Distillation

Three core modules:
  - CAGS: Class-Adaptive Guidance Strength
  - IAST: IPC-Adaptive Stop Timing
  - TAGS: Timestep-Adaptive Guidance Scheduling
"""

from .class_complexity import ClassComplexityAnalyzer
from .adaptive_stop import AdaptiveStopTiming
from .guidance_schedule import TimestepAdaptiveSchedule
from .ags_sampler import AGSSampler

__all__ = [
    "ClassComplexityAnalyzer",
    "AdaptiveStopTiming",
    "TimestepAdaptiveSchedule",
    "AGSSampler",
]

__version__ = "0.1.0"