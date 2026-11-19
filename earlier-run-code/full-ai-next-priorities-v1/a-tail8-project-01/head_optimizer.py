"""Complete disjoint ownership: retained fine weights and all added coarse weights."""
import math

def groups(network,learning_rate):
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Positive learning rate required')
    head=[p for name,p in network.named_parameters() if name=='coarse_logits' or name.startswith('shape_head.')]
    fine=[p for name,p in network.named_parameters() if name!='coarse_logits' and not name.startswith('shape_head.')]
    if not head or not fine or len({id(p) for p in head+fine})!=len(list(network.parameters())):
        raise ValueError('Disjoint complete optimizer ownership required')
    return [dict(params=fine,lr=learning_rate,lr_multiplier=1.),
            dict(params=head,lr=learning_rate*(10/3),lr_multiplier=(10/3))]
