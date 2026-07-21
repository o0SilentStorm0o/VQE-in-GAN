"""Training primitives for auditable ACGAN experiments."""

from .diagnostics import (
    GradientDiagnostics,
    measure_gradient_diagnostics,
)
from .steps import (
    DiscriminatorStepMetrics,
    GeneratorStepMetrics,
    discriminator_step,
    generator_step,
)

__all__ = [
    "DiscriminatorStepMetrics",
    "GradientDiagnostics",
    "GeneratorStepMetrics",
    "discriminator_step",
    "generator_step",
    "measure_gradient_diagnostics",
]
