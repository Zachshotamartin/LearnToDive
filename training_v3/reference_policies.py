"""Immutable, physically compatible baselines for model-selection evidence."""
import hashlib
import io
from pathlib import Path

import torch

from checkpointing import hashes
from policy import Policy, FORMAT, MOTOR_FORMAT, FINAL_FORMAT


def load_reference(path, expected_digest=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_digest and digest != expected_digest:
        raise ValueError('Reference checkpoint changed: ' + str(path))
    saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    contract, config = saved['contract'], saved['config']
    if contract['format'] not in (FORMAT, MOTOR_FORMAT, FINAL_FORMAT, 'self-declared-diver-state-covariance-v1'):
        raise ValueError('Unsupported reference policy format')
    current = hashes()
    for name in ('diver.xml', 'geometry.py', 'stance.py', 'water.py'):
        if contract['sourceHashes'].get(name) != current[name]:
            raise ValueError('Reference physical model changed: ' + name)
    # Constructing a read-only evaluator must not advance training randomness.
    with torch.random.fork_rng():
        model = Policy(contract['observationSize'], config['widths'], config['rho'],
                       exploration=config.get('exploration', 'diagonal'),
                       architecture=config.get('architecture', 'shared'),
                       noise_rho=config.get('noise_rho', 0.), normalize_inputs=config.get('input_normalization', False))
    model.load_state_dict(saved['model'])
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Non-finite reference weights')
    model.eval().requires_grad_(False)
    return model, dict(path=str(Path(path).resolve()), sha256=digest, steps=saved['training']['steps'])
