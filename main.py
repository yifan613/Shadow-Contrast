from src.config import load_config
from src.runner import Runner
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    runner = Runner(config)
    runner.test()
    print('testing complete')
if __name__ == "__main__":
    main()