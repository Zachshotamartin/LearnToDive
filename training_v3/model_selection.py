"""Independent development champions and conservative replacement of the incumbent."""
from collections import defaultdict
import math
from assessment_stats import paired_interval, gate

CHAMPIONS = ('points', 'execution', 'clean')
# Measured units: judged execution points, proportions, and degrees.
GUARDS = dict(execution=.1, clean=.02, jumped=.05, valid=.02, alignment=2.,
              entryArms=.02, positionQuality=.03, motorPositionQuality=.03, completedRotation=.02)
MINIMUM_POINTS_GAIN = .5


def measurements(report, category=None):
    grouped = defaultdict(list)
    for row in report['episodes']:
        if not row['practice'] and (category is None or row['category'] == category):
            grouped[str(row['index'])].append(row)
    values = {}
    for key in ('points', *GUARDS):
        def value(row):
            if key == 'jumped':
                return float(row['measurements']['takeoffVerticalSpeed'] > .5)
            if key == 'alignment':
                return -row['entryAngle']
            if key == 'entryArms':
                return float(row['measurements']['entryArmPositionValid'])
            if key in ('positionQuality', 'motorPositionQuality'):
                return float(row['measurements'][key])
            if key == 'completedRotation':
                from judge import ROTATION_TOLERANCE, TWIST_TOLERANCE
                return float(row['rotationError'] <= ROTATION_TOLERANCE and row['twistError'] < TWIST_TOLERANCE)
            return float(row[key])
        values[key] = {i: sum(value(row) for row in rows) / len(rows) for i, rows in grouped.items()}
    return values


def comparison(candidate, reference):
    for key in ('seed', 'cases', 'deterministic', 'policySeed', 'perturbed', 'evaluationVersion'):
        if candidate.get(key) != reference.get(key):
            raise ValueError('Evaluation contract changed: ' + key)
    a, b = measurements(candidate), measurements(reference)
    # Bonferroni intervals across the primary outcome and every guard metric.
    confidence = 1 - .05 / len(a)
    contrasts = {k: paired_interval(a[k], b[k], confidence=confidence) for k in a}
    result = gate(contrasts, GUARDS, 'points', MINIMUM_POINTS_GAIN)
    # A pooled improvement cannot hide a material category regression.
    category_reasons = []
    for category in range(1, 7):
        x, y = measurements(candidate, category), measurements(reference, category)
        for metric, tolerance in GUARDS.items():
            if x[metric] and y[metric]:
                delta = sum(x[metric].values()) / len(x[metric]) - sum(y[metric].values()) / len(y[metric])
                if delta < -tolerance:
                    category_reasons.append(f'category {category} {metric}: observed regression')
            elif bool(x[metric]) != bool(y[metric]):
                category_reasons.append(f'category {category}: coverage changed')
    result['reasons'].extend(category_reasons)
    result['eligibleForReview'] = not result['reasons']
    return dict(**result, contrasts=contrasts, unit='whole six-dive routine/world', scope='reused development cohort')


def competence(report):
    """Absolute full-dive requirements; practice success cannot qualify a model."""
    reasons = []
    for category, row in report['summary']['categories'].items():
        if row['n'] < 5:
            reasons.append(f'category {category}: insufficient full-dive coverage')
            continue
        for metric, minimum in dict(clean=.5, valid=.8, entryArms=.8, motorPositionQuality=.6, completedRotation=.8).items():
            if row[metric] < minimum:
                reasons.append(f'category {category}: {metric} below {minimum}')
        if category != '6' and (row['jumped'] < .8 or row['rise'] < .15):
            reasons.append(f'category {category}: insufficient genuine takeoff')
    return dict(qualified=not reasons, reasons=reasons, automaticPublication=False)


def update_selection(state, report):
    """Return checkpoint labels to write. All evidence stays in resumable state."""
    metrics = report['summary']['full']
    if any(metrics[k] is None or not math.isfinite(metrics[k]) for k in CHAMPIONS):
        raise ValueError('Cannot select from missing or non-finite judged metrics')
    book = state.setdefault('selection', dict(version=1, champions={}, incumbent=None))
    labels = []
    for name in CHAMPIONS:
        old = book['champions'].get(name)
        if old is None or metrics[name] > old['value']:
            book['champions'][name] = dict(value=metrics[name], steps=report['steps'])
            labels.append('best-' + name)
    old = book['incumbent']
    decision = (dict(eligibleForReview=True, reasons=['first evaluated development baseline'], automaticPublication=False)
                if old is None else comparison(report, old))
    reference_decisions = [comparison(report, reference) for reference in state.get('referenceReports', [])]
    book['referenceComparisons'] = reference_decisions
    book['competence'] = competence(report) if 'categories' in report['summary'] else dict(qualified=False, reasons=['missing categories'])
    book['eligibleForFinalTest'] = (book['competence']['qualified']
                                   and bool(reference_decisions)
                                   and all(r['eligibleForReview'] for r in reference_decisions))
    book['lastDecision'] = dict(steps=report['steps'], **decision)
    if decision['eligibleForReview']:
        book['incumbent'] = report
        state['bestValue'] = metrics['points']
        labels.append('best')
    return labels
