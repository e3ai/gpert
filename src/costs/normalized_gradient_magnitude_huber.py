import logging


from . import GradientMagnitudeHuber, NormalizedGradientMagnitude

logger = logging.getLogger(__name__)


class NormalizedGradientMagnitudeHuber(NormalizedGradientMagnitude):
    """Normalized gradient magnitude
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "normalized_gradient_magnitude_huber"
    required_keys = ["orig_iwe", "iwe", "omit_boundary"]

    def __init__(
        self,
        direction="minimize",
        store_history: bool = False,
        cuda_available=False,
        precision="32",
        *args,
        **kwargs,
    ):
        super().__init__(
            direction=direction,
            store_history=store_history,
            cuda_available=cuda_available,
            precision=precision,
        )
        # Overwrite using Huber version.
        self.gradient_magnitude = GradientMagnitudeHuber(
            direction=direction,
            store_history=store_history,
            cuda_available=cuda_available,
            precision=precision,
        )
