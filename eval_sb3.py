"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import argparse
import os
import gym
import pandas as pd
from env.custom_hopper import *
import torch

import matplotlib.pyplot as plt

from time import gmtime, strftime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.evaluation import evaluate_policy

def positive_int(x):
   x = int(x)
   if x <= 0:
      raise argparse.ArgumentTypeError("Needs to be positive")
   return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, help='Training environment [source, target, source-udr]', required=True)
    parser.add_argument('--n-episodes', default=50, type=positive_int, help='Number of episodes to evaluate the agent')
    parser.add_argument('--env', default='source', type=str, help='Training environment [source, target, source-udr]')
    parser.add_argument('--logs-models-path', default='logs_and_models', type=str, help='Path to the logs_and_models directory')
    parser.add_argument('--model-name', default='best_model', type=str, help='Name of the model in the run directory')

    return parser.parse_args()

args = parse_args()

def main():
      envs = {
         "source":"CustomHopper-source-v0",
         "target":"CustomHopper-target-v0",
         "source-udr":"CustomHopper-source-UDR-v0"
         }
      run_dir = os.path.join(args.logs_models_path, args.run_id)

      eval_env = gym.make(envs[args.env])
      eval_env = Monitor(eval_env, os.path.join(run_dir, "eval_monitor.csv"))

      model_path = os.path.join(run_dir, args.model_name)

      model = PPO.load(model_path)

      eval_records_path = os.path.join(run_dir, "eval_records.csv")
      try:
          eval_records = pd.read_csv(eval_records_path)
      except OSError:
          print("No existing evaluation records found for this run. Creating new records.")
          eval_records = pd.DataFrame(columns=["model", "returns_mean", "returns_std"])

      returns_mean, returns_std = evaluate_policy(model, eval_env, n_eval_episodes=args.n_episodes)
      
      print(f"mean: {returns_mean}")
      print(f"std: {returns_std}")
      
      new_row = pd.DataFrame([[args.model_name, returns_mean, returns_std]], columns=["model", "returns_mean", "returns_std"])
      eval_records = pd.concat([eval_records, new_row], ignore_index=True)
      eval_records.to_csv(eval_records_path, index=False)

if __name__ == '__main__':
    main()