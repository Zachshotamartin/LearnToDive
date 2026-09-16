"""Mastery-gated scope: what the athlete may declare, how random its stance is, how it practises.

Every stage is the same self-declaration game with a smaller legal set. The
learner still chooses its own dive within the stage; nothing prescribes a
motion. A stage widens only after the held-out clean rate on its own scope has
stayed above the threshold for several consecutive evaluations, so the final
scope (every group, height and apparatus at full randomisation) is the finish
line rather than the starting line.
"""
import numpy as np

ADVANCE_CLEAN_RATE = .3
ADVANCE_EVALUATIONS = 3
ALL_GROUPS = (1, 2, 3, 4, 5, 6)
STAGES = (
    dict(name='platform 10 m, forward and back', apparatus=(1,), platformHeights=(10,), springboardHeights=(),
         groups=(1, 2), maxTurns=1., maxTwists=0., positions=('A', 'C'), spread=.25, auxiliary=.6),
    dict(name='inward and pike', apparatus=(1,), platformHeights=(10,), springboardHeights=(),
         groups=(1, 2, 4), maxTurns=1., maxTwists=0., positions=('A', 'B', 'C'), spread=.5, auxiliary=.5),
    dict(name='reverse and one and a half', apparatus=(1,), platformHeights=(10,), springboardHeights=(),
         groups=(1, 2, 3, 4), maxTurns=1.5, maxTwists=0., positions=('A', 'B', 'C'), spread=.75, auxiliary=.4),
    dict(name='all platform heights and twists', apparatus=(1,), platformHeights=(5, 7.5, 10), springboardHeights=(),
         groups=(1, 2, 3, 4, 5), maxTurns=1.5, maxTwists=1., positions=('A', 'B', 'C', 'D'), spread=1., auxiliary=.3),
    dict(name='armstand', apparatus=(1,), platformHeights=(5, 7.5, 10), springboardHeights=(),
         groups=ALL_GROUPS, maxTurns=1.5, maxTwists=1., positions=('A', 'B', 'C', 'D'), spread=1., auxiliary=.3),
    dict(name='springboard', apparatus=(0, 1), platformHeights=(5, 7.5, 10), springboardHeights=(1, 3),
         groups=ALL_GROUPS, maxTurns=1.5, maxTwists=1., positions=('A', 'B', 'C', 'D'), spread=1., auxiliary=.25),
)
FINAL_STAGE = len(STAGES) - 1


def stage(index):
    if not 0 <= int(index) <= FINAL_STAGE:
        raise ValueError('Unknown mastery stage')
    return STAGES[int(index)]


def allows(scope, dive):
    """Whether a declaration lies inside a stage's scope (armstand turns count double)."""
    limit = scope['maxTurns'] * (2. if dive['armstand'] else 1.)
    return (dive['group'] in scope['groups'] and dive['turns'] <= limit
            and dive['twists'] <= scope['maxTwists'] and dive['position'] in scope['positions'])


def scope_mask(scope, dives):
    return np.array([allows(scope, d) for d in dives], bool)


def routine_groups(scope, apparatus):
    """The groups a routine on this apparatus cycles through, in table order."""
    groups = [g for g in scope['groups'] if apparatus == 1 or g != 6]
    return groups


def heights_for(scope, apparatus):
    return scope['platformHeights'] if apparatus == 1 else scope['springboardHeights']


class Progression:
    """Consecutive-evaluation counter that decides when a stage widens."""

    def __init__(self, index=0):
        self.index = int(index)
        self.consecutive = 0
        self.history = []

    def observe(self, clean_rate, steps):
        """Record one evaluation; returns True when the stage just advanced."""
        if clean_rate is not None and clean_rate >= ADVANCE_CLEAN_RATE:
            self.consecutive += 1
        else:
            self.consecutive = 0
        if self.consecutive >= ADVANCE_EVALUATIONS and self.index < FINAL_STAGE:
            self.history.append(dict(fromStage=self.index, toStage=self.index + 1, steps=int(steps), cleanRate=float(clean_rate)))
            self.index += 1
            self.consecutive = 0
            return True
        return False

    def state_dict(self):
        return dict(index=self.index, consecutive=self.consecutive, history=list(self.history))

    def load_state_dict(self, state):
        self.index = int(state['index'])
        self.consecutive = int(state['consecutive'])
        self.history = list(state['history'])
