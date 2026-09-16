"""Atomic JSON and checkpoint writes plus the exact-resume source contract."""
import hashlib
import json
import os
from pathlib import Path

import torch

from engine import TRAINING_SOURCES

HERE = Path(__file__).resolve().parent


def hashes():
    """SHA-256 of every file that defines the learning problem.

    Only those files are part of the exact-resume contract; editing the suite
    controller or the audit scripts must not strand a paused run.
    """
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in TRAINING_SOURCES}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)
