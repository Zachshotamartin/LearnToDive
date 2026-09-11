"""Register a frozen fixed-case evaluation; does not qualify or publish assets."""
import argparse,json
from checkpoints import register_evaluation
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('evaluation');p.add_argument('--checkpoint');a=p.parse_args()
 print(json.dumps({'bestUpdated':register_evaluation(a.run,a.evaluation,a.checkpoint)}))
