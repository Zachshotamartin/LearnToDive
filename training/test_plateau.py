import copy
import unittest
from train_until_plateau import plateau, metrics


class PlateauTests(unittest.TestCase):
    def history(self, count=11):
        return [{'additionalSteps': i * 2048000,
                 'metrics': {s: {'clean': 0., 'execution': 1., 'alignment': -30., 'geometry': .1}
                             for s in ['all', 'forward', 'back', 'reverse', 'inward', 'twist', 'pike']}}
                for i in range(count)]

    def test_minimum_and_patience(self):
        self.assertFalse(plateau(self.history(10), 20480000, 5))
        self.assertFalse(plateau(self.history(5), 1, 5))
        self.assertTrue(plateau(self.history(), 20480000, 5))

    def test_one_improving_dive_prevents_stop(self):
        h = self.history()
        h[-2]['metrics']['twist']['alignment'] += .6
        self.assertFalse(plateau(h, 20480000, 5))

    def test_regression_then_recovery_is_not_new_progress(self):
        h = self.history()
        for row in h[-5:-1]: row['metrics']['all']['execution'] = .2
        self.assertTrue(plateau(h, 20480000, 5))

    def test_continued_slow_progress_accumulates(self):
        h = self.history()
        for i, row in enumerate(h): row['metrics']['forward']['execution'] += i * .03
        self.assertFalse(plateau(h, 20480000, 5))

    def test_invalid_report_fails(self):
        r = {'policies': {'pretrained': dict(cleanCompletionRate=0, meanExecution=1,
             meanEntryAngle=30, geometryPassRate=.1)}, 'perSkill': {'pretrained': {}}}
        with self.assertRaises(ValueError): metrics(r)
        r['perSkill']['pretrained'] = {str(i): copy.deepcopy(r['policies']['pretrained']) for i in range(6)}
        self.assertEqual(len(metrics(r)), 7)
        r['perSkill']['pretrained']['0']['meanExecution'] = float('nan')
        with self.assertRaises(ValueError): metrics(r)


if __name__ == '__main__': unittest.main()
