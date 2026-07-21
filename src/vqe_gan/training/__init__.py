"""Training primitives for auditable ACGAN experiments."""

from .diagnostics import (
    GradientDiagnostics,
    measure_distribution_gradient_diagnostics,
    measure_gradient_diagnostics,
)
from .steps import (
    DiscriminatorStepMetrics,
    GeneratorStepMetrics,
    coherence_guided_generator_step,
    discriminator_step,
    distribution_regularized_generator_step,
    generator_step,
)

__all__ = [
    "DiscriminatorStepMetrics",
    "GradientDiagnostics",
    "GeneratorStepMetrics",
    "coherence_guided_generator_step",
    "discriminator_step",
    "distribution_regularized_generator_step",
    "generator_step",
    "measure_distribution_gradient_diagnostics",
    "measure_gradient_diagnostics",
]
