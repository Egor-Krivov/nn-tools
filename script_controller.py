#!/bin/python
import argparse
import time
import subprocess

if __name__ == '__main__':

    # Parsing
    parser = argparse.ArgumentParser('Script controller')
    parser.add_argument('script', type=str, help='script path')

    parser.add_argument('-a', '--attempts', type=int, default=100,
                        help='maximum number of script restarts')

    args = parser.parse_args()

    script = args.script
    attempts = args.attempts

    try:
        for attempt in range(attempts):
            start = time.time()
            r = subprocess.run(['python', script]).returncode
            end = time.time()
            print('return code is {}, time {}'.format(r, end-start))
            if r == 0:
                break

    except KeyboardInterrupt:
        print('\nInterrupted by keyboard')
