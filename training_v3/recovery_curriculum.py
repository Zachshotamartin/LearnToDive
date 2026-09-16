"""Diverse recovery starts collected exclusively from the learner's own flights."""
import copy
import numpy as np

CAPACITY_PER_GROUP = 512
UNIFORM_SHARE = .25
RECENCY_SCALE = 5000


def recovery_stage(height_ratio):
    return 0 if height_ratio > .65 else 1 if height_ratio > .3 else 2


class RecoveryArchive:
    def __init__(self):
        self.groups = [[] for _ in range(6)]
        self.next_id = 0
        self.clock = 0
        self.readiness = np.zeros((6, 3))

    def add(self, group, snapshot, features, stage):
        self.clock += 1
        bucket = tuple(np.floor(np.asarray(features) * [5, 4, 4, 4]).astype(int))
        key = (snapshot['declaration'], snapshot['height'], snapshot['apparatus'], *bucket)
        rows = self.groups[group]
        # Refresh redundant physical states, preserving measured usefulness.
        existing = next((row for row in rows if row['key'] == key), None)
        if existing is not None:
            existing['snapshot'] = copy.deepcopy(snapshot)
            return
        row = dict(id=self.next_id, key=key, snapshot=copy.deepcopy(snapshot), stage=stage,
                   fast=.0, slow=.0, visits=0, last=self.clock)
        self.next_id += 1
        rows.append(row)
        if len(rows) > CAPACITY_PER_GROUP:
            # Remove a representative from the most crowded stage/height stratum.
            strata = [(r['stage'], r['snapshot']['height'], r['snapshot']['apparatus']) for r in rows]
            crowded = max(set(strata), key=lambda key: (strata.count(key), key))
            candidates = [r for r, key in zip(rows, strata) if key == crowded]
            victim = min(candidates, key=lambda r: (abs(r['fast'] - r['slow']), r['last'], r['id']))
            rows.remove(victim)

    def choose(self, group, declaration, height, apparatus, rng):
        rows = [r for r in self.groups[group] if r['snapshot']['declaration'] == declaration
                and r['snapshot']['height'] == height and r['snapshot']['apparatus'] == apparatus]
        if not rows:
            return None
        self.clock += 1
        readiness = self.readiness[group]
        # Start with later recoveries; earlier states become more frequent as later
        # entry control improves. A nonzero floor keeps every available stage reachable.
        stage_weights = [.1 + readiness[1], .25 + readiness[2], 1.]
        scores = np.array([stage_weights[r['stage']] * (.1 + abs(r['fast'] - r['slow'])
                           + .15 / np.sqrt(1 + r['visits'])
                           + .1 * min(1, (self.clock - r['last']) / RECENCY_SCALE)) for r in rows])
        probabilities = (1 - UNIFORM_SHARE) * scores / scores.sum() + UNIFORM_SHARE / len(rows)
        row = rows[int(rng.choice(len(rows), p=probabilities))]
        row['last'] = self.clock
        return row

    def observe(self, group, identity, quality):
        quality = float(np.clip(quality, 0, 1))
        row = next((r for r in self.groups[group] if r['id'] == identity), None)
        if row is None:
            return
        row['visits'] += 1
        row['fast'] += .2 * (quality - row['fast'])
        row['slow'] += .03 * (quality - row['slow'])
        stage = row['stage']
        self.readiness[group, stage] += .05 * (quality - self.readiness[group, stage])

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        self.__dict__.update(copy.deepcopy(state))
