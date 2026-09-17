"""Milestones measure a run and are recorded; they never stop it."""
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_suite
from train import GATES, REVIEW_PHASES, Trainer


def report(entry_success, clean_rate, clean_dives):
    """The fields a milestone reads out of a held-out evaluation."""
    return dict(summary=dict(full=dict(clean=clean_rate)),
                episodes=[dict(clean=True, practice=False)] * clean_dives,
                motorSkills=dict(tasks=dict(entry=dict(success=entry_success))))


class MilestoneTests(unittest.TestCase):
    def trainer(self, steps):
        trainer = Trainer.__new__(Trainer)
        trainer.state = dict(steps=steps)
        trainer.args = SimpleNamespace(gates=1)
        return trainer

    def test_a_missed_milestone_records_its_reasons_and_is_measured_once(self):
        trainer = self.trainer(GATES[0]['steps'])
        missed = trainer.check_gates(report(.29, 0., 0))
        self.assertFalse(missed['passed'])
        self.assertEqual(len(missed['reasons']), 2)
        self.assertEqual([row['name'] for row in trainer.state['gates']], [GATES[0]['name']])
        # A later evaluation at the same milestone does not measure it again.
        self.assertIsNone(trainer.check_gates(report(.9, .5, 3)))
        self.assertEqual(len(trainer.state['gates']), 1)

    def test_a_met_milestone_records_that_it_passed(self):
        trainer = self.trainer(GATES[0]['steps'])
        self.assertIsNone(trainer.check_gates(report(.75, .1, 2)))
        self.assertTrue(trainer.state['gates'][0]['passed'])
        self.assertEqual(trainer.state['gates'][0]['reasons'], [])

    def test_a_milestone_is_not_measured_before_its_step_count(self):
        trainer = self.trainer(GATES[0]['steps'] - 1)
        self.assertIsNone(trainer.check_gates(report(0., 0., 0)))
        self.assertEqual(trainer.state.get('gates', []), [])

    def test_no_phase_or_control_path_ends_a_run_on_a_milestone(self):
        for phase in REVIEW_PHASES + run_suite.FINISHED_PHASES:
            self.assertNotIn('gate', phase)
        for name in ('train.py', 'run_suite.py'):
            self.assertNotIn('gate-failed', Path(__file__).with_name(name).read_text(), name)


if __name__ == '__main__':
    unittest.main()
