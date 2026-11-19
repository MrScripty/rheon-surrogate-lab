"""One complete disjoint tiny-model parameter group."""
import math
def groups(network,learning_rate):
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError("Positive finite rate required")
    return [dict(params=list(network.parameters()),lr=learning_rate,lr_multiplier=1.)]
