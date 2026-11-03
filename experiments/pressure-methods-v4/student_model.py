"""Student fine network with the full model's learned global branch retained."""
import copy
import json
from pathlib import Path
import teacher_model
from student_slice import slice_width

ARCHITECTURE={};WIDTH=None
def __getattr__(name):return getattr(teacher_model,name)

def select(config):
    global WIDTH
    teacher_model.select(config)
    spec=json.loads(Path(__file__).with_name('distillation.json').read_text());WIDTH=tuple(spec['width_fraction'])
    ARCHITECTURE.clear();ARCHITECTURE.update(copy.deepcopy(teacher_model.ARCHITECTURE))
    ARCHITECTURE.update(encoder_widths=[n*WIDTH[0]//WIDTH[1]for n in [8,16,24,32]],
        decoder_widths=[n*WIDTH[0]//WIDTH[1]for n in [32,24,16,8]],bottleneck_width=48*WIDTH[0]//WIDTH[1],activation='linear')
    ARCHITECTURE['student_distillation']=dict(width_fraction=list(WIDTH),teacher_weights_sha256=spec['teacher_weights_sha256'],
        distillation_weight=spec['weight'],target='Teacher fine correction on each student-generated inner pressure residual',
        teacher_used_at_inference=False,normalization='Shared fixed original C53 scales')

class PressureCorrectionNet(teacher_model.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        if WIDTH is None:raise ValueError('Select the captured student width first')
        sliced,_=slice_width(self,*WIDTH);self._modules=sliced._modules;self.requires_grad_(True)
