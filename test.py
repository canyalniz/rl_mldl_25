"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import os

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy, ValueEstimator
import pandas as pd

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
    parser.add_argument('--logs-models-path', default='logs_and_models', type=str, help='Path to the logs_and_models directory')
    parser.add_argument('--model-name', default='best_model', type=str, help='Name of the model in the run directory')
    parser.add_argument('--print-only', action='store_true', help='If this flag is set this evaluation run will not be recorded in the global eval_records.csv in the logs_and_models directory')

    return parser.parse_args()

args = parse_args()


def main():
    envs = {
         "source":"CustomHopper-source-v0",
         "target":"CustomHopper-target-v0",
         "source-udr":"CustomHopper-source-UDR-v0"
         }
    run_dir = os.path.join(args.logs_models_path, args.run_id)

    env = gym.make(envs[args.eval_env])

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    model_name = args.model_name
    
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(os.path.join(run_dir, model_name + "_policy.mdl")), strict=True)

    value_function = None
    if "critic" in model_name:
        value_function = ValueEstimator(observation_space_dim)
        value_function.load_state_dict(torch.load(os.path.join(run_dir, model_name + "_value_function.mdl")), strict=True)

    agent = Agent(policy=policy, value_function=value_function, run_id=args.run_id)

    rewards = np.zeros(args.n_episodes)
    for episode in range(args.n_episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:

            action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            test_reward += reward

        rewards[episode] = test_reward
    
    returns_mean = np.mean(rewards)
    returns_std = np.std(rewards)

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