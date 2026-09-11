"""Native-versus-JavaScript inference parity on real observations; no publication."""
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from environment import Arena
from policy import Policy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOLERANCE = 1e-5
SCRIPT = '''import fs from 'node:fs';
import {createAutonomousPolicy} from %s;
const f = JSON.parse(fs.readFileSync(process.argv[1]));
const p = createAutonomousPolicy(f.model);
let error = 0;
for (let i = 0; i < f.observations.length; i++) {
  const r = p.predict(f.observations[i], f.masks[i]);
  if (r.choice !== f.choices[i]) throw Error('Choice mismatch');
  for (let j = 0; j < 9; j++) error = Math.max(error, Math.abs(r.action[j] - f.expected[i][j]));
}
if (error > %s) throw Error('Motor mismatch ' + error);
console.log(JSON.stringify({maximumAbsoluteError: error, cases: f.observations.length}));
'''


def fixture(e):
    """A random untrained policy, real observations and the native deterministic outputs."""
    p = Policy(e.observation_size, (96, 96))
    o = torch.tensor(e.observe())
    mask = torch.tensor(e.mask())
    x = p(o, mask, torch.tensor(e.choosing), deterministic=True)
    return dict(model=p.export(), observations=o.tolist(), masks=mask.tolist(), expected=x['action'].tolist(),
                choices=x['choice'].tolist())


def main():
    torch.set_num_threads(1)
    torch.manual_seed(817)
    e = Arena(8, seed=442, training=False)
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            path.write_text(json.dumps(fixture(e)))
            script = SCRIPT % (json.dumps((ROOT / 'src/core/autonomousPolicy.js').as_uri()), TOLERANCE)
            subprocess.run(['node', '--input-type=module', '-e', script, str(path)], check=True)
    finally:
        e.close()


if __name__ == '__main__':
    main()
