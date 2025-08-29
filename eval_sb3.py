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

from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy

def positive_int(x):
   x = int(x)
   if x <= 0:
      raise argparse.ArgumentTypeError("Needs to be positive")
   return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, help='ID corresponding to the run to identify the run directory', required=True)
    parser.add_argument('--n-episodes', default=50, type=positive_int, help='Number of episodes to evaluate the agent')
    parser.add_argument('--eval-env', default='source', type=str, help='Training environment [source, target, source-udr]')
    parser.add_argument('--train-env', default='source', type=str, help='Training environment [source, target, source-udr]')
    parser.add_argument('--logs-models-path', default='logs_and_models', type=str, help='Path of the logs_and_models directory')
    parser.add_argument('--eval-logs-folder', default='eval_monitor', type=str, help='Name of the evaluation logs directory inside the logs_and_models directory')
    parser.add_argument('--model-name', default='best_model', type=str, help='Name of the model in the run directory')
    parser.add_argument('--print-only', action='store_true', help='If this flag is set this evaluation run will not be recorded in the global eval_records.csv in the logs_and_models directory')

    return parser.parse_args()

args = parse_args()

def main():
      envs = {
         "source":"CustomHopper-source-v0",
         "target":"CustomHopper-target-v0",
         "source-udr":"CustomHopper-source-UDR-v0",
         "source-normal":"CustomHopper-normal-UDR-v0"
         }
      run_dir = os.path.join(args.logs_models_path, args.run_id)
      model_name = args.model_name


      eval_env = gym.make(envs[args.eval_env])
      eval_logs_path = os.path.join(run_dir, args.eval_logs_folder, model_name + "_monitor.csv")
      eval_env = Monitor(eval_env, eval_logs_path)

      model_path = os.path.join(run_dir, model_name)

      model = PPO.load(model_path)

      
      returns_mean, returns_std = evaluate_policy(model, eval_env, n_eval_episodes=args.n_episodes)
      
      print(f"mean: {returns_mean}")
      print(f"std: {returns_std}")
      
      if not args.print_only:
        eval_records_path = os.path.join(args.logs_models_path, "eval_records.csv")
        try:
            eval_records = pd.read_csv(eval_records_path)
        except OSError:
            print("No existing global evaluation records found. Creating new records.")
            eval_records = pd.DataFrame(columns=["run_id", "eval_env", "train_env", "returns_mean", "returns_std"])

        new_row = pd.DataFrame([[args.run_id, args.eval_env, args.train_env, returns_mean, returns_std]], columns=["run_id", "eval_env", "train_env", "returns_mean", "returns_std"])
        eval_records = pd.concat([eval_records, new_row], ignore_index=True)
        eval_records.to_csv(eval_records_path, index=False)

if __name__ == '__main__':
    main()