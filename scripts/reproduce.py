"""Rebuild/check all defined experiments from the packaged data, without downloads.

python scripts/reproduce.py --train-tabular
Omit --train-tabular to keep the selected tabular run; both modes re-evaluate it.
Identical completed runs are integrity-checked and reused.
"""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train-tabular',action='store_true');args=parser.parse_args()
    steps=[]
    if args.train_tabular:steps.append(['-m','experiments.tabular','train','--task','all'])
    steps.extend([['-m','experiments.images','evaluate','--dataset','all'],['-m','experiments.sequences'],
                  ['-m','experiments.rag'],['-m','experiments.agents'],['scripts/benchmark_rl.py'],
                  ['scripts/verify_submission.py','--with-llm']])
    for step in steps:
        print('\nRUN:', ' '.join(step),flush=True)
        subprocess.run([sys.executable,'-X','utf8','-u',*step],cwd=ROOT,check=True)


if __name__=='__main__':main()
