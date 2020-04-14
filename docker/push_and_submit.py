import subprocess
import argparse
import random
import sys
import csv

def get_experiment_id():
    with open('helper/words.csv', 'r') as f:
        words = csv.reader(f)
        wordlist = list(words)
    random.shuffle(wordlist)
    experiment_id = '{0}_{1}'.format(wordlist[0][0], wordlist[1][0])
    print('Experiment ID: {0}'.format(experiment_id))
    return experiment_id

parser = argparse.ArgumentParser(description='Training run utility')
parser.add_argument('--exp_id', type=str, default=None, help='Experiment ID')
parser.add_argument('--model_dir', type=str, default='tx_model', help='Experiment ID')
args = parser.parse_args()

if not args.exp_id:
    experiment_id = get_experiment_id()
else:
    experiment_id = args.exp_id

subprocess.run(["./push_training_container.sh", experiment_id, args.model_dir])
subprocess.run(["./submit_training_run.sh", experiment_id])
