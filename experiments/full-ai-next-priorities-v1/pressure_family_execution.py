"""Apply the same qualified learned fine execution policy across pressure models."""
from precise_routed_tail_pressure import PreciseRoutedTailPrediction
from precise_routed_baseline_pressure import PreciseRoutedBaselinePrediction


def create(state):
    return PreciseRoutedTailPrediction(state)if hasattr(state.network,'tail_logits')else PreciseRoutedBaselinePrediction(state)
