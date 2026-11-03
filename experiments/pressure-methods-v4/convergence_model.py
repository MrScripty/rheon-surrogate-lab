"""Full-width neural proposal with two GCR iterations for training and export."""
import copy
import teacher_model
import fine_tuning

ARCHITECTURE={}
def __getattr__(name):return getattr(teacher_model,name)
def select(config):
    teacher_model.select(config)
    ARCHITECTURE.clear();ARCHITECTURE.update(copy.deepcopy(teacher_model.ARCHITECTURE))
    ARCHITECTURE['solver_training']=dict(method='Two differentiable flexible GCR iterations, double orthogonalization, original true residual',
        proposal='One original learned-coarse and fine-network correction',deployment='Two GCR iterations; external diagnostic policies may extend the same proposal',
        objective='Canonical physical loss plus matched final and early log equation-residual feedback')
PressureCorrectionNet=teacher_model.PressureCorrectionNet
def predict_pressure(network,sample,normalization):return fine_tuning.predict(network,sample,normalization,teacher_model.owner,2)
def training_loss(network,sample,normalization,criterion=None):return fine_tuning.loss(teacher_model,network,sample,normalization,criterion)
