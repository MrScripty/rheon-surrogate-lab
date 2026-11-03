"""Finite CUDA training with diagnostics over fitting parents only.

The captured trainer owns optimizer, model, loss, custody and restore parity.
This entrypoint replaces its validation diagnostic with
an explicitly identified diagnostic over original training parents only.
"""
import math
import train


def diagnostic(network, parents, normalization):
    train.require(parents and all(p["entry"]["split"] == "train" for p in parents),
                  "diagnostic_population", "original training parents only")
    network.eval()
    grouped = []
    with train.torch.no_grad():
        for parent in parents:
            values = []
            for frame in range(len(parent["frames"])):
                sample = train.data.sample(parent, frame, device=next(network.parameters()).device)
                loss = train.model.pressure_loss(train.model.predict_pressure(network, sample, normalization),
                                                sample, normalization)
                train.require(bool(train.torch.isfinite(loss).all()), "diagnostic_nonfinite", "training loss")
                values.append(float(loss.mean()))
            grouped.append(math.fsum(values) / len(values))
    return math.fsum(grouped) / len(grouped)


original_write = train.write


def write(path, value):
    if isinstance(value, dict) and ("validation" in value or value.get("schema") == "rheon-water-learning-training-v1"):
        value = {**value, "diagnostic_population": "train_only", "heldout_evaluations": 0,
                 "scientific_acceptance_or_model_promotion": False}
    original_write(path, value)


if __name__ == "__main__":
    train.evaluate = diagnostic
    train.write = write
    train.main()
