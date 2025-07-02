"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def main():
    # # train_env = gym.make('CustomHopper-source-v0')

    # Parallel environments
    vec_env = make_vec_env("CustomHopper-source-v0", n_envs=8, vec_env_cls=SubprocVecEnv)

    print('State space:', vec_env.observation_space)  # state-space
    print('Action space:', vec_env.action_space)  # action-space
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

    # model = PPO(env=vec_env, **hyperparams)
    model = PPO(policy="MlpPolicy", env=vec_env, verbose=1, device="cpu")
    model.learn(total_timesteps=5e6, progress_bar=True)
    model.save("ppo_hopper_5e6")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_hopper_5e6")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

if __name__ == '__main__':
    main()