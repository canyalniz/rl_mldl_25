"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import argparse
import os
import gym
from env.custom_hopper import *

from stable_baselines3 import PPO

def positive_int(x):
   x = int(x)
   if x <= 0:
      raise argparse.ArgumentTypeError("Needs to be positive")
   return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, help='ID corresponding to the run to identify the run directory', required=True)
    parser.add_argument('--env', default='source', type=str, help='Training environment [source, target, source-udr]')
    parser.add_argument('--logs-models-path', default='logs_and_models', type=str, help='Path to the logs_and_models directory')
    parser.add_argument('--model-name', default='best_model', type=str, help='Path to the logs_and_models directory')

    return parser.parse_args()

args = parse_args()

def main():
    envs = {
       "source":"CustomHopper-source-v0",
       "target":"CustomHopper-target-v0",
       "source-udr":"CustomHopper-source-UDR-v0"
       }
    run_directory = os.path.join(args.logs_models_path, args.run_id)

    env = gym.make(envs[args.env])

    model_path = os.path.join(run_directory, args.model_name)

    model = PPO.load(model_path)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")

if __name__ == '__main__':
    main()