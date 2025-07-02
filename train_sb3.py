"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import os
import gym
from env.custom_hopper import *
import torch

import matplotlib.pyplot as plt

from time import gmtime, strftime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import results_plotter
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.evaluation import evaluate_policy

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
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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

def main():
    # Unique tag of this run based on the timestamp
    run_tag = strftime("%Y-%m-%d--%H_%M_%S", gmtime())

    # Create log dir belonging to this run
    log_dir = os.path.join("logs_and_models", run_tag)
    os.makedirs(log_dir, exist_ok=False)

    # Create a vector environment
    n_envs = 8
    vec_env = make_vec_env("CustomHopper-source-v0", n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    # Use the Monitor wrapper to record experiment results
    vec_env = VecMonitor(vec_env, log_dir)

    # print('State space:', vec_env.observation_space)  # state-space
    # print('Action space:', vec_env.action_space)  # action-space
    # print('Dynamics parameters:', vec_env.get_parameters())  # masses of each link of the Hopper

    hyperparams = dict(
            # env_wrapper=[{"gymnasium.wrappers.TimeLimit": {"max_episode_steps": 100}}],
            normalize_advantage=True,
            # n_envs=1,
            policy="MlpPolicy",
            batch_size=32,
            n_steps=512,
            gamma=0.999,
            learning_rate=9.80828e-05,
            ent_coef=0.00229519,
            clip_range=0.2,
            n_epochs=5,
            gae_lambda=0.99,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            # use_sde=True,
            policy_kwargs=dict(
                log_std_init=-2,
                ortho_init=False,
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            ),
            verbose=1,
            device="cpu",
    )

    timesteps=2e6

    # Create instance of the callback that saves the best model
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, skip_over=2e5, log_dir=log_dir)

    # model = PPO(env=vec_env, **hyperparams)
    model = PPO(policy="MlpPolicy", env=vec_env, verbose=1, device="cpu")
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)

    # del model # remove to demonstrate saving and loading

    # model = PPO.load(os.path.join(log_dir, "best_model"))

    # print(evaluate_policy(model, vec_env, n_eval_episodes=n_envs*2))

    # obs = vec_env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     vec_env.render("human")

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO-Hopper")
    plt.show()

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

if __name__ == '__main__':
    main()