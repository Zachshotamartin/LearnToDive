"""Versioned legal declarations from the official difficulty table.

The complete table keeps checkpoint indices stable. The practice/preview mask
also limits complexity; the official difficulty table and judge stay intact.
"""
import json
from pathlib import Path

import numpy as np

DATA = json.loads(Path(__file__).with_name('difficulty.json').read_text())
GROUPS = ('forward', 'backward', 'reverse', 'inward', 'twisting', 'armstand')
POSITIONS = ('A', 'B', 'C', 'D')
TWISTING_GROUP = 5
ARMSTAND_GROUP = 6


def decode(row):
    """Expand a difficulty-table row into rotation, direction and entry facts, or None."""
    code = str(row['code'])
    group = int(code[0])
    direction = int(code[1]) if group in (TWISTING_GROUP, ARMSTAND_GROUP) else group
    if group < TWISTING_GROUP:
        if code[1] != '0':
            return None  # Flying-position rules need a separate phase judge.
        halves = int(code[2:])
        twists = 0
    else:
        halves = int(code[2])
        twists = int(code[3]) if len(code) > 3 else 0
    if direction not in range(1, 5):
        return None
    armstand = group == ARMSTAND_GROUP
    headfirst = (halves % 2 == 0) if armstand else (halves % 2 == 1)
    return dict(row, group=group, direction=direction, halves=halves, twists=twists / 2, turns=halves / 2,
                sign=1 if direction in (1, 2) else -1, back=int(direction in (2, 4)), armstand=armstand,
                headfirst=headfirst)


DIVES = [d for row in DATA['rows'] if (d := decode(row)) is not None]
IDS = {r['id']: i for i, r in enumerate(DIVES)}
CODES = sorted({r['code'] for r in DIVES})
CODE_INDEX = {c: i for i, c in enumerate(CODES)}


def table_key(apparatus, height):
    return f'{apparatus}:{height:g}'


def difficulty(index, apparatus, height):
    row = DIVES[int(index)]
    key = table_key(apparatus, height)
    if key not in row['difficulty']:
        raise ValueError('No official difficulty for this apparatus/height')
    return row['difficulty'][key]


PRACTICE_SCOPE = 'max-1.5-flips-2-armstand-1-twist-v1'


def in_practice_scope(d):
    return d['turns'] <= (2. if d['armstand'] else 1.5) and d['twists'] <= 1.


def legal_mask(group, apparatus, height, used=()):
    """Boolean mask over DIVES of declarations legal in this round."""
    key = table_key(apparatus, height)
    used = set(used)
    return np.array([in_practice_scope(d) and d['group'] == group and key in d['difficulty'] and d['code'] not in used for d in DIVES], bool)


def validate_declaration(index, group, apparatus, height, used=()):
    if not 0 <= int(index) < len(DIVES) or not legal_mask(group, apparatus, height, used)[int(index)]:
        raise ValueError('Illegal category, apparatus, position, or repeated dive declaration')
    return DIVES[int(index)]
