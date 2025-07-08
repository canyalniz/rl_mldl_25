"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import os
from time import gmtime, strftime

from stable_baselines3 import PPO
import torch
import gym
import tqdm

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-name', default="best_model", type=str, help='Name of model that should be used to collect transition data')
	parser.add_argument('--run-id', default="source_optuna_2", type=str, help='ID corresponding to the run to identify the run directory')
	parser.add_argument('--env', default="target", type=str, help='The environment for which the transition data should be collected [source, target]')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--logs-models-path', default='logs_and_models', type=str, help='Path to the logs_and_models directory')
	parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
	parser.add_argument('--episodes', default=100, type=int, help='Number of episodes to collect data over')
	parser.add_argument('--output-dir', default='dropo_datasets/transitions', type=str, help='Path to the transitions directory')

	return parser.parse_args()

args = parse_args()


def main():
	envs = {
       "source":"CustomHopper-source-v0",
       "target":"CustomHopper-target-v0",
       }
	
	env = gym.make(envs[args.env])

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	model_name = args.model_name
	model_path = os.path.join(args.logs_models_path, args.run_id, model_name)
	model = PPO.load(model_path)

	collections = {
		"observations" : [],
		"next_observations" : [],
		"actions" : [],
		"terminals" : []
		}

	for _ in tqdm.tqdm(range(args.episodes)):
		done = False
		state = env.reset()

		while not done:

			action, _ = model.predict(state)
			previous_state = state

			state,_,done,_ = env.step(action)
			
			collections["observations"].append(previous_state)
			collections["next_observations"].append(state)
			collections["actions"].append(action)
			collections["terminals"].append(done)

			if args.render:
				env.render()
	
	for k,v in collections.items():
		collection_name = strftime("%Y-%m-%d--%H_%M_%S_", gmtime()) + model_name + "_" + k
		save_path = os.path.join(args.output_dir, collection_name)
		np.save(save_path, np.array(v))

if __name__ == '__main__':
	main()