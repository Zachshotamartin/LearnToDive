"""Versioned physical/control semantics carried by every new checkpoint.
The XML hash is checked independently of byte integrity in the public manifest.
"""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
JUDGE_VERSION='full-submersion-physical-entry-v8'
CONTROL_SEMANTICS='joint-servo-full-entry-v8'

def runtime_contract(xml_path=None):
 return dict(schemaVersion=1,physics='MuJoCo 3.13.0',geometrySemantics='shared-head-and-full-depth-basin-v8.1',waterVersion='directional-submerged-body-v1',xmlSHA256=hashlib.sha256(Path(xml_path or ROOT/'public/physics/diver.xml').read_bytes()).hexdigest(),controlSemantics=CONTROL_SEMANTICS,judgeVersion=JUDGE_VERSION,observationSemantics='upright-adduction-full-entry-v4',observationSize=76,actionSize=9,actionLow=[-.45,0,-.55,-.5,-.5,-1.1,-1.1,0,-.15],actionHigh=[2.2,2.6,1.35,3.14,3.14,1.1,1.1,2.3,.18],footModel='combined ankle and midfoot pitch; physical range -0.6 to 1.4 rad, target maximum 1.35 rad, unchanged 100 Nm torque cap')

def validate_contract(actual,expected=None):
 expected=expected or runtime_contract()
 if not isinstance(actual,dict):raise ValueError('Checkpoint has no physical/control contract; legacy weights require an explicit cross-physics baseline migration.')
 for key in ['schemaVersion','physics','waterVersion','xmlSHA256','controlSemantics','judgeVersion','observationSemantics','observationSize','actionSize','actionLow','actionHigh']:
  if actual.get(key)!=expected.get(key):raise ValueError(f'Checkpoint/runtime contract mismatch: {key}')
 return actual
