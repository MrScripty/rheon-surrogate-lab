"""Same learned coarse inference; declare a separate rate for newly added logits."""
import copy
import learned_coarse_model

ARCHITECTURE = {}


def __getattr__(name):
    return getattr(learned_coarse_model,name)


def select(config):
    learned_coarse_model.select(config)
    ARCHITECTURE.clear()
    ARCHITECTURE.update(copy.deepcopy(learned_coarse_model.ARCHITECTURE))
    ARCHITECTURE['coarse_optimizer_followup'] = dict(
        same_initialization_and_data=True,fine_learning_rates=[1e-5,1e-6,1e-7],
        coarse_learning_rates=[.01,.001,.0001],coarse_parameters=63,
        same_shared_loss_and_global_gradient_clip=True,
        rationale='The matched-rate arm barely changes new logits; isolate optimization of the new branch')
