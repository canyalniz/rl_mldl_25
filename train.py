"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import os
from time import gmtime, strftime, time

import torch
import gym
import tqdm

from env.custom_hopper import *
from agent import Agent, Policy, ValueEstimator

def positive_int(x):
   x = int(x)
   if x <= 0:
      raise argparse.ArgumentTypeError("Needs to be positive")
   return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', default=1000000, type=positive_int, help='Number of training timesteps')
    parser.add_argument('--check-freq', default=1000, type=positive_int, help='Wait for this many timesteps before checking to see if you should save the model')
    parser.add_argument('--skip-over', default=750000, type=positive_int, help='Wait for this many timesteps before checking to see if you should save the model')
    parser.add_argument('--device', default='cpu', type=str, help='Network device [cpu, cuda]')
    parser.add_argument('--env', default='source', type=str, help='Training environment [source, target, source-udr]')
    parser.add_argument('--model-name', default='best_model', type=str, help='Model will be saved by this name in the run directory corresponding to the id')
    parser.add_argument('--run-id', default=None, type=str, help='ID of the run, if no id is provided it is automatically assigned according to the timestamp')
    parser.add_argument('--print-after-n-episodes', default=1000, type=positive_int, help="How many episodes to wait between printing status information")
    parser.add_argument('--critic', default=False, action="store_true", help="Option to activate Actor-Critic")
    parser.add_argument('--baseline-vf', default=False, action="store_true", help="Option to activate the value function baseline for REINFORCE")
    parser.add_argument('--baseline', default=0, type=int, help='Scalar baseline value to use for REINFORCE')

    return parser.parse_args()

args = parse_args()


def main():
    # Unique tag of this run based on the timestamp
    run_tag = strftime("%Y-%m-%d--%H_%M_%S", gmtime()) if args.run_id is None else args.run_id

    # Create log dir belonging to this run
    run_dir = os.path.join("logs_and_models", run_tag)
    os.makedirs(run_dir, exist_ok=True)

    if os.path.exists(os.path.join(run_dir, args.model_name + "_policy.mdl")):
       raise FileExistsError("The model already exists, refusing to overwrite, exiting...")

    envs = {
       "source":"CustomHopper-source-v0",
       "target":"CustomHopper-target-v0",
       "source-udr":"CustomHopper-source-UDR-v0",
       "source-normal":"CustomHopper-source-normal-v0"
       }
    env = gym.make(envs[args.env])

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())


    """
        Training
    """
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    
    value_function = None
    if args.critic:
        value_function = ValueEstimator(observation_space_dim)
    
    agent = Agent(policy=policy, run_id=args.run_id, value_function=value_function, critic=args.critic, device=args.device, model_name=args.model_name, check_freq=args.check_freq, skip_over=args.skip_over, scalar_baseline=args.baseline)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

    timestep = 0
    episode = 0
    t_start = time()
    try:
        agent.set_metadata(t_start=t_start, env_id=envs[args.env])
        with tqdm.tqdm(total=args.timesteps) as pbar:
            while timestep <= args.timesteps:
                done = False
                train_reward = 0
                state = env.reset()  # Reset the environment and observe the initial state

                while not done:  # Loop until the episode is over

                    action, action_probabilities = agent.get_action(state)
                    previous_state = state

                    state, reward, done, info = env.step(action.detach().cpu().numpy())

                    agent.store_outcome(previous_state, state, action_probabilities, reward, done, time() - t_start)

                    train_reward += reward

                    if args.critic:
                        agent.update_policy()
                    
                    timestep += 1
                    pbar.update(1)

                if not args.critic:
                    agent.update_policy()
                
                episode += 1
                
                if episode%args.print_after_n_episodes == 0:
                    print('Training episode:', episode)
                    print('Episode return:', train_reward)
        
        agent.save_monitor_csv()
    except KeyboardInterrupt as e:
        agent.save_model()
        agent.save_monitor_csv()
        
        

    

if __name__ == '__main__':
    main()