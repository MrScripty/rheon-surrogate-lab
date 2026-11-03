"""Create only the declared tiny model initialization; never train here."""
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import model,data,torch
model.configure('cpu');config=json.load(open(Path(sys.argv[1])/'config.json'));model.select(config);torch.manual_seed(config['training']['seed'])
network=model.PressureCorrectionNet()
inventory=data.helpers().save_weights(network,Path(sys.argv[2]))
print(json.dumps(inventory))
