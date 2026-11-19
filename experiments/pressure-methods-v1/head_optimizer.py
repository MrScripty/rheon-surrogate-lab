"""Keep C53 fine-weight rates fixed while allowing new coarse gains to learn."""
import math

MULTIPLIER = 1000.


def groups(network,learning_rate):
    if not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Positive finite learning rate required')
    head = network.coarse_logits
    if tuple(head.shape)!=(63,) or not head.requires_grad:
        raise ValueError('Exactly 63 trainable coarse logits required')
    fine = [value for name,value in network.named_parameters() if name!='coarse_logits']
    if not fine or {id(p)for p in fine}|{id(head)} != {id(p)for p in network.parameters()} or id(head)in {id(p)for p in fine}:
        raise ValueError('Complete disjoint fine/head parameter ownership required')
    return [dict(params=fine,lr=learning_rate,lr_multiplier=1.),
            dict(params=[head],lr=learning_rate*MULTIPLIER,lr_multiplier=MULTIPLIER)]
