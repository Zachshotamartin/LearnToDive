"""Versioned legal declarations. Targets are chosen by the policy, never the trainer."""
import json
from pathlib import Path
import numpy as np
DATA=json.loads(Path(__file__).with_name('difficulty.json').read_text())
GROUPS=('forward','backward','reverse','inward','twisting','armstand')
POSITIONS=('A','B','C','D')

def decode(row):
 c=str(row['code']); group=int(c[0]); direction=int(c[1]) if group in (5,6) else group
 if group<5:
  if c[1]!='0':return None # Flying-position rules need a separate phase judge.
  halves=int(c[2:]);twists=0
 else:halves=int(c[2]);twists=int(c[3]) if len(c)>3 else 0
 if direction not in range(1,5):return None
 arm=group==6; head=(halves%2==0) if arm else (halves%2==1)
 return dict(row,group=group,direction=direction,halves=halves,twists=twists/2,
   turns=halves/2,sign=1 if direction in (1,2) else -1,back=int(direction in (2,4)),
   armstand=arm,headfirst=head)
DIVES=[d for row in DATA['rows'] if (d:=decode(row)) is not None]
IDS={r['id']:i for i,r in enumerate(DIVES)}
CODES=sorted({r['code'] for r in DIVES});CODE_INDEX={c:i for i,c in enumerate(CODES)}

def difficulty(index, apparatus, height):
 row=DIVES[int(index)];key=f'{apparatus}:{height:g}'
 if key not in row['difficulty']:raise ValueError('No official difficulty for this apparatus/height')
 return row['difficulty'][key]

def legal_mask(group,apparatus,height,used=()):
 key=f'{apparatus}:{height:g}';used=set(used)
 return np.array([d['group']==group and key in d['difficulty'] and d['code'] not in used for d in DIVES],bool)

def validate_declaration(index,group,apparatus,height,used=()):
 if not 0<=int(index)<len(DIVES) or not legal_mask(group,apparatus,height,used)[int(index)]:
  raise ValueError('Illegal category, apparatus, position, or repeated dive declaration')
 return DIVES[int(index)]
