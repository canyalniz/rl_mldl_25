"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import argparse
import os
import gym
import yaml
from env.custom_hopper import *

import matplotlib.pyplot as plt

from time import gmtime, strftime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from modified_sb3.common.callbacks import EvalCallback

def positive_int(x):
   x = int(x)
   if x <= 0:
      raise argparse.ArgumentTypeError("Needs to be positive")
   return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', default=1600000, type=positive_int, help='Number of training timesteps')
    parser.add_argument('--eval-freq', default=1000, type=positive_int, help='Timesteps between evaluating the model')
    parser.add_argument('--eval-wait', default=800000, type=positive_int, help='Wait for this many timesteps before starting to evaluate the model')
    parser.add_argument('--n-envs', default=8, type=positive_int, help='Number of environments to run in parallel in the vector environment')
    parser.add_argument('--device', default='cpu', type=str, help='Network device [cpu, cuda]')
    parser.add_argument('--env', default='source', type=str, help='Training environment [source, target, source-udr, source-normal]')
    parser.add_argument('--eval-env', default='source', type=str, help='Evaluation environment [source, target, source-udr, source-normal]')
    parser.add_argument('--run-id', default=None, type=str, help='ID of the run, if no id is provided it is automatically assigned according to the timestamp')

    return parser.parse_args()

args = parse_args()

def main():
    
    # Unique tag of this run based on the timestamp
    run_tag = strftime("%Y-%m-%d--%H_%M_%S", gmtime()) if args.run_id is None else args.run_id

    # Create log dir belonging to this run
    run_dir = os.path.join("logs_and_models", run_tag)
    
    if os.path.exists(run_dir):
       raise FileExistsError("The run already exists, refusing to overwrite, exiting...")
    
    os.makedirs(run_dir, exist_ok=True)

    ## CREATE TRAINING ENVIRONMENT ##
    n_envs = args.n_envs
    # Mapping from command-line input to source tag
    envs = {
       "source":"CustomHopper-source-v0",
       "target":"CustomHopper-target-v0",
       "source-udr":"CustomHopper-source-UDR-v0",
       "source-normal":"CustomHopper-source-normal-v0"
       }

    # Path to the monitor csv to be created by Monitor Wrapper
    train_log_filename = os.path.join(run_dir, "train_monitor.csv")
    if n_envs > 1:
      # Create a vector environment
      train_env = make_vec_env(envs[args.env], n_envs=n_envs, vec_env_cls=SubprocVecEnv)
      # Use the VecMonitor wrapper to record experiment results
      train_env = VecMonitor(train_env, train_log_filename)
    else:
      # Create single vector environment
      train_env = gym.make(envs[args.env])
      # Use the Monitor wrapper to record experiment results
      train_env = Monitor(train_env, train_log_filename)

    ## SET HYPERPARAMETERS ##
    hyperparams={
       'learning_rate': 0.0006029621422146909,
       'n_steps': 2**11,
       'gamma': 1-0.009293230682418586,
       'gae_lambda': 1 - 0.06027683642684977,
       'ent_coef': 7.94158163281704e-06,
       'clip_range': 0.14778195849652764,
       'vf_coef': 0.5083057178072663,
       'policy': 'MlpPolicy',       
       }

    timesteps = args.timesteps
    
    ## SAVE HYPERPARAMETERS ##
    # Create the model without initializing it, this is to get hyperparameter information
    model = PPO(env=train_env, device=args.device, verbose=1, _init_setup_model=False, **hyperparams)
    # Get argument names to PPO.__init__
    PPO_init_arguments = PPO.__init__.__code__.co_varnames[:PPO.__init__.__code__.co_argcount]
    # Get the hyperparameter values used to initialize the model
    hyperparams_reference = {arg:model.__dict__[arg] for arg in PPO_init_arguments if arg not in ["self", "_init_setup_model", "policy", "env"]}
    hyperparams_reference['device'] = str(hyperparams_reference['device'])
    # Save hyperparameters for reference
    with open(os.path.join(run_dir, 'hyperparams.yaml'), 'w') as f:
       yaml.dump(hyperparams_reference, f)
    
    ## MODEL CREATION ##
    # Create the model according to the given command-line arguments and the given hyperparameters
    model = PPO(env=train_env, device=args.device, verbose=1, _init_setup_model=True, **hyperparams)
    
    ## TRAINING ##    
    if n_envs > 1:
      # Create a vector environment
      eval_env = make_vec_env(envs[args.eval_env], n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    else:
      # Create single vector environment
      train_env = gym.make(envs[args.eval_env])

    eval_callback = EvalCallback(eval_env, best_model_save_path=run_dir,
                             log_path=run_dir, eval_freq = max(args.eval_freq // n_envs, 1),
                             wait_for=max(args.eval_wait // n_envs, 1), deterministic=True, render=False)
    # Learn the policy
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=eval_callback)

    ## PLOTTING ##
    plot_results([run_dir], timesteps, results_plotter.X_TIMESTEPS, args.run_id)
    plt.show()


if __name__ == '__main__':
    main()