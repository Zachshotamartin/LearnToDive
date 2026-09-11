from pathlib import Path
import json,re,hashlib
import pymupdf as f
import argparse
a=argparse.ArgumentParser();a.add_argument('pdf');args=a.parse_args()
root=Path(__file__).resolve().parent
p=f.open(args.pdf);rows={}
for apparatus,pages,heights in [('springboard',range(163,166),[1,3]),('platform',range(168,173),[10,7.5,5])]:
 for i in pages:
  for table in p[i].find_tables().tables:
   for cells in table.extract():
    if len(cells)==4+4*len(heights):cells=[cells[0], ' '.join(x for x in cells[1:4] if x)]+cells[4:]
    if not cells[0] or not re.fullmatch(r'[1-6]\d{2,3}',cells[0].strip()) or len(cells)!=2+4*len(heights):continue
    code=cells[0].strip()
    for j,cell in enumerate(cells[2:]):
     if not cell or not re.fullmatch(r'\d\.\d',cell.strip()):continue
     pos='ABCD'[j%4];key=f'{code}{pos}'
     rec=rows.setdefault(key,dict(id=key,code=int(code),position=pos,name=cells[1].replace('\n',' '),difficulty={}))
     k=f'{apparatus}:{heights[j//4]:g}'
     v=float(cell)
     if k in rec['difficulty'] and rec['difficulty'][k]!=v:raise ValueError((key,k))
     rec['difficulty'][k]=v
catalog=dict(source='https://resources.fina.org/fina/document/2026/02/18/e6815ecc-06d9-4f0b-98e9-4c441cf5e6a3/2026-02-18_World-Aquatics_CR-Final.pdf',sourceSHA256=hashlib.sha256(Path(args.pdf).read_bytes()).hexdigest(),version='World Aquatics February 2026, appendices 9 and 11',rows=sorted(rows.values(),key=lambda x:x['id']))
(root/'difficulty.json').write_text(json.dumps(catalog,indent=2)+'\n')
for k in ['101C','103B','203C','5132D','6243D']:print(k,rows.get(k))
print(len(rows),'dive/position combinations',sum(len(x['difficulty']) for x in rows.values()),'DD values')
