"""Regression guards, honest cluster statistics and actual simulator evaluation."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from assessment_stats import paired_interval
from comparison_summary import summarize_comparison
from model_selection import update_selection, ranking, outranks, regressed
from train import Trainer, parser
from evaluation import evaluate
from assess_behavior import assess
from compare_rewards import arms_for


def report(points=10, execution=5, clean=.5, step=1):
    rows=[]
    for i in range(8):
        for category in range(1,7):
            rows.append(dict(index=i,category=category,practice=False,points=points,execution=execution,clean=clean,
                valid=1.,entryAngle=10.,rotationError=0.,twistError=0.,
                measurements=dict(takeoffVerticalSpeed=2.,entryArmPositionValid=True,positionQuality=.8,motorPositionQuality=.8)))
    return dict(seed=77,cases=8,deterministic=True,policySeed=1,perturbed=False,evaluationVersion=2,steps=step,
                summary=dict(full=dict(points=points,execution=execution,clean=clean)),episodes=rows)


class AssessmentTests(unittest.TestCase):
    def test_best_follows_the_suite_ranking_not_the_incumbent_gate(self):
        state = {}
        self.assertIn('best', update_selection(state, report(10, 5, .5, 1)))
        # A regression keeps the champion; the per-metric books still advance.
        self.assertNotIn('best', update_selection(state, report(3, 1, .5, 2)))
        self.assertEqual(state['selection']['champion']['steps'], 1)
        # Below the incumbent gate's minimum gain, yet ranked higher: best moves on.
        labels = update_selection(state, report(10.2, 5, .5, 3))
        self.assertIn('best', labels)
        self.assertEqual(state['selection']['champion']['steps'], 3)
        self.assertEqual(state['bestValue'], 10.2)
        self.assertFalse(state['selection']['lastDecision']['eligibleForReview'])
        self.assertEqual(state['selection']['incumbent']['steps'], 1)
        # Clean dives outrank any amount of judged points.
        self.assertIn('best', update_selection(state, report(1, .2, .6, 4)))
        self.assertEqual(state['selection']['champion']['ranking'][:3], [.6, .2, 1.])

    def test_ranking_and_regression_against_the_champion(self):
        champion = dict(steps=8, ranking=list(ranking(dict(clean=0., execution=2., points=12.))), points=12.)
        self.assertEqual(ranking(dict(clean=0., execution=2., points=12.)), (0., 2., 12., 0., 0.))
        self.assertTrue(outranks(dict(clean=.05, execution=.1, points=1.), champion))
        self.assertFalse(outranks(dict(clean=0., execution=2., points=12.), champion))
        self.assertTrue(outranks(dict(clean=0., execution=2., points=12.), None))
        self.assertTrue(regressed(dict(points=5.9), champion))
        self.assertFalse(regressed(dict(points=6.), champion))
        self.assertFalse(regressed(dict(points=0.), None))
        self.assertFalse(regressed(dict(points=0.), dict(steps=2, ranking=[0, 0, 0, 0, 0], points=0.)))

    def test_champions_preserve_clean_model_when_points_regress_execution(self):
        state={}
        self.assertIn('best',update_selection(state,report()))
        labels=update_selection(state,report(10.01,4,.3,2))
        self.assertEqual(labels,['best-points'])
        self.assertEqual(state['selection']['incumbent']['steps'],1)
        self.assertTrue(state['selection']['lastDecision']['reasons'])
        labels=update_selection(state,report(12,5,.5,3))
        self.assertIn('best',labels)
        clone=copy.deepcopy(state)
        self.assertEqual(update_selection(clone,report(12,5,.5,4)),[])

    def test_pair_validation_and_seed_uncertainty(self):
        with self.assertRaises(ValueError):paired_interval({'a':1},{'b':2})
        with self.assertRaises(ValueError):paired_interval({'a':float('nan')},{'a':2})
        a={str(i):1+i/10 for i in range(6)}; b={k:v-.5 for k,v in a.items()}
        result=paired_interval(a,b)
        self.assertAlmostEqual(result['difference'],.5)
        self.assertGreater(result['interval'][0],0)
        self.assertTrue(paired_interval({'a':1},{'a':0})['preliminary'])
        rows=[dict(seed=i,metrics={k:dict(points=p,execution=5,clean=.5,jumped=.5,valid=.8) for k,p in [('baseline',10),('better',12)]}) for i in range(5)]
        self.assertTrue(summarize_comparison(rows,'baseline')['comparisons']['better']['eligibleForReview'])
        self.assertFalse(summarize_comparison(rows[:3],'baseline')['comparisons']['better']['eligibleForReview'])
        with self.assertRaises(ValueError):summarize_comparison(rows+[rows[0]],'baseline')

    def test_native_modes_are_repeatable_restore_rng_and_save_phase_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            args=parser().parse_args(['--output',folder,'--envs','2','--threads','1','--widths','16','16','--steps','0'])
            trainer=Trainer(args)
            try:
                trainer.persist('test')
                before=torch.get_rng_state().clone()
                for deterministic in (True,False):
                    a=evaluate(trainer.policy,seed=771101,cases=2,deterministic=deterministic,policy_seed=13)
                    b=evaluate(trainer.policy,seed=771101,cases=2,deterministic=deterministic,policy_seed=13)
                    self.assertEqual(a,b)
                    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
                    self.assertEqual(len(a['episodes']),12)
                    self.assertEqual(a['summary']['practiceEpisodes'],0)
                    self.assertIn('postContactRotationDegrees',a['summary']['full'])
                result=assess(Path(folder)/'latest.pt',[881101],cases=1)
                self.assertFalse(result['automaticPublication'])
                self.assertEqual(len(result['evaluations']),2)
            finally:trainer.env.close()

    def test_native_checkpoint_champions_are_saved_with_resumable_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            args=parser().parse_args(['--output',folder,'--envs','2','--threads','1','--widths','16','16','--steps','0','--eval-cases','2'])
            trainer=Trainer(args)
            try:
                trainer.run_evaluation()
                for name in ('best','best-points','best-execution','best-clean'):
                    saved=torch.load(Path(folder)/(name+'.pt'),weights_only=False)
                    self.assertEqual(saved['training']['selection']['incumbent']['steps'],0)
                    self.assertIn('optimizer',saved)
            finally: trainer.env.close()

    def test_fresh_and_continued_have_identical_hyperparameters(self):
        arms=dict(arms_for('initialization'))
        self.assertEqual(arms['fresh'],arms['continued'])
