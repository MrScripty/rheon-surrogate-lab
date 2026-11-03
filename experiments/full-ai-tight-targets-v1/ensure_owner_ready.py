"""Use the already-authorized saved owner workflow, logging in the new study."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/full-ai-next-priorities-v1'))
import owner_ready
owner_ready.BASE=ROOT/'runs/full-ai-tight-targets-v1'
if __name__=='__main__':owner_ready.run()
