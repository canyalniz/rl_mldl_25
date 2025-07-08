import optuna
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor

from env.custom_hopper import *

def make_vec_train_env():
    # Create a vector environment
    train_env = make_vec_env("CustomHopper-source-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
    # Use the Monitor wrapper to record experiment results
    train_env = VecMonitor(train_env)
    return train_env

def make_env():
    return Monitor(gym.make("CustomHopper-source-v0"))

def optimize_ppo(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3)
    n_steps = 2**trial.suggest_int("n_steps_exponent", 9,12)
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.01, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.005, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    vf_coef = trial.suggest_float("vf_coef", 0.45, 0.9, log=True)

    train_env = make_vec_train_env()
    test_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log=None,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        vf_coef=vf_coef,
        device="cpu",
    )

    # Short training for quick evaluation
    model.learn(total_timesteps=int(8e5))
    train_env.close()

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, test_env, n_eval_episodes=5, deterministic=True)
    test_env.close()

    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="ppo_source_study",
        direction="maximize",
        storage="sqlite:///optuna_studies/ppo_source_study.db",
        load_if_exists=True
    )

    try:
        study.optimize(optimize_ppo, n_trials=13, gc_after_trial=True, show_progress_bar=True)
    except KeyboardInterrupt:
        pass
    

    print("Best trial:")
    print(study.best_trial)
