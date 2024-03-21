import argparse
import datetime
import subprocess

if __name__ == "__main__":
    current_date = datetime.datetime.now().date()
    print(f'Current date: {current_date}')

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--verbose', help='show everything', action='store_true')
    args = argument_parser.parse_args()

    if args.verbose:
        print(f'Using verbose mode')

    subprocess.run(['jupyter nbconvert','--to python SequencePredictionNew.ipynb'])