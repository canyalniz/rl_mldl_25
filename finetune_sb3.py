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
from stable_baselines3.common.callbacks import BaseCallback

def positive_int(x):
   x = int(x)
   if x <= 0:
      raise argparse.ArgumentTypeError("Needs to be positive")
   return x

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("lr not a floating-point literal")

    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("lr not in range (0.0, 1.0]")
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', default=800000, type=positive_int, help='Number of training timesteps')
    parser.add_argument('--check-freq', default=1000, type=positive_int, help='Wait for this many timesteps before checking to see if you should save the model')
    parser.add_argument('--skip-over', default=0, type=positive_int, help='Wait for this many timesteps before checking to see if you should save the model')
    parser.add_argument('--n-envs', default=8, type=positive_int, help='Number of environments to run in parallel')
    parser.add_argument('--device', default='cpu', type=str, help='Network device [cpu, cuda]')
    parser.add_argument('--env', default='source', type=str, help='Training environment [source, target, source-udr]')
    parser.add_argument('--run-id', default=None, type=str, required=True, help='ID of the run, if no id is provided it is automatically assigned according to the timestamp')
    parser.add_argument('--model-name', default='best_model', type=str, help='Name of the model in the run directory')
    parser.add_argument('--iter-code', default="0", type=str, help='Fine tuning iteration you are running on the same starting model')
    parser.add_argument('--lr-factor', default=0.55, type=restricted_float, help='The original lr will be multiplied by this factor before the linear lr decay starts')

    return parser.parse_args()

args = parse_args()
new_model_name = args.model_name + args.iter_code

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param skip_over: Wait this many timesteps before evaluating models
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, skip_over: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.skip_over = skip_over
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, new_model_name)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.n_calls>=self.skip_over) and (self.n_calls % self.check_freq == 0):

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def main():
    # Unique tag of this run based on the timestamp
    run_tag = args.run_id

    # Create log dir belonging to this run
    run_dir = os.path.join("logs_and_models", run_tag)
    if not os.path.isdir(run_dir):
       raise ValueError("The run directory does not exist.")

    n_envs = args.n_envs
    envs = {
       "source":"CustomHopper-source-v0",
       "target":"CustomHopper-target-v0",
       "source-udr":"CustomHopper-source-UDR-v0"
       }

    log_filename = os.path.join(run_dir, new_model_name + "_monitor.csv")
    if n_envs > 1:
      # Create a vector environment
      train_env = make_vec_env(envs[args.env], n_envs=n_envs, vec_env_cls=SubprocVecEnv)
      # Use the Monitor wrapper to record experiment results
      train_env = VecMonitor(train_env, log_filename)
    else:
      train_env = gym.make(envs[args.env])
      train_env = Monitor(train_env, log_filename)

    try:
       with open(os.path.join(run_dir, "hyperparams.yaml")) as f:
          hyperparams = yaml.safe_load(f)
    except OSError as e:
       hyperparams = {}
       # hardcoding the default lr to provide the starting point for the decay
       hyperparams["learning_rate"] = 0.0003
       print("Warning: Could not access hyperparameter file, using default hyperparameters.")
    
    timesteps = args.timesteps

    # Create instance of the callback that saves the best model
    callback = SaveOnBestTrainingRewardCallback(check_freq=args.check_freq, skip_over=(args.skip_over // n_envs), log_dir=run_dir)

    # model = PPO(env=vec_env, **hyperparams)
    model_path = os.path.join(run_dir, args.model_name)
    model = PPO.load(
       model_path,
       env=train_env,
       device=args.device,
       custom_objects={
          "learning_rate" : linear_schedule(hyperparams["learning_rate"] * args.lr_factor),
        }
      )

    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)

    plot_results([run_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO-Hopper")
    plt.savefig(os.path.join(run_dir, new_model_name + "training_graph.png"), format="png")
    plt.show()


if __name__ == '__main__':
    main()