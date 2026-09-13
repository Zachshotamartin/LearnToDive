"""Compare matched training seeds; checkpoint episodes are not independent runs."""
import argparse
import json
from pathlib import Path
from assessment_stats import paired_interval, gate

METRICS = ('points', 'execution', 'clean', 'jumped', 'valid')
GUARDS = {'execution': 0.1, 'clean': 0.02, 'jumped': 0.05, 'valid': 0.02}
PRIMARY = 'points'
MIN_GAIN = 0.5


def summarize_comparison(rows, baseline):
    if not rows or len({str(r['seed']) for r in rows}) != len(rows):
        raise ValueError('Use one result per independent training seed')
    arms = set(rows[0]['metrics'])
    if baseline not in arms or len(arms) < 2 or any(set(r['metrics']) != arms for r in rows):
        raise ValueError('All training seeds must contain the same baseline and candidates')
    confidence = 1 - .05 / ((len(arms) - 1) * len(METRICS))
    result = {}
    for arm in sorted(arms - {baseline}):
        contrasts = {}
        for metric in METRICS:
            a,b = [{str(r['seed']):r['metrics'][name][metric] for r in rows} for name in (arm, baseline)]
            contrasts[metric] = paired_interval(a,b,confidence=confidence,minimum_units=5)
        result[arm] = dict(contrasts=contrasts, **gate(contrasts,GUARDS,PRIMARY,MIN_GAIN))
    return dict(baseline=baseline, comparisons=result, independentUnit='training seed',
        scope='Equal-new-experience paired development comparison. Inherited lifetime experience differs for fresh versus continued.',
        multiplicity='Bonferroni-adjusted intervals over all declared arm/metric comparisons',
        finalHoldoutRequired=True, automaticPublication=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists(): raise ValueError('Preserve completed statistics')
    data=json.loads(Path(a.results).read_text())
    out.write_text(json.dumps(summarize_comparison(data['pairedSeeds'],a.baseline),indent=2,allow_nan=False)+'\n')
