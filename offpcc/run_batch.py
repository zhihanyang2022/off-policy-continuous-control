import concurrent.futures
from subprocess import Popen, check_output
from argparse import ArgumentParser
import os
import yaml
from threading import Lock

def run(cmd):
    p = Popen(cmd, shell=True)
    p.wait()

def run_job(cmd):
    print(cmd, flush=True)
    run(cmd)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default="commands.yaml")
    parser.add_argument('--n-workers', type=int, default=2)
    args = parser.parse_args()

    with open(args.file) as f:
        content = f.readlines()

    cmds = [x.strip() for x in content] 

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        for cmd in cmds:
            if not cmd.startswith("#") and not cmd.startswith(" "):
                executor.submit(run_job, cmd)
