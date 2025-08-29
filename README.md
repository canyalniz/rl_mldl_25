# Project on Reinforcement Learning (Course project MLDL 2025 - POLITO)
### Teaching assistants: Andrea Protopapa and Davide Buoso

Repository of group 64 for "Project 4: Reinforcement Learning" course project of MLDL 2025 at Polytechnic of Turin. Official assignment at [Google Doc](https://docs.google.com/document/d/16Fy0gUj-HKxweQaJf97b_lTeqM_9axJa4_SdqpP_FaE/edit?usp=sharing).

This project repository builds on top of the [official template repository](https://github.com/lambdavi/rl_mldl_25).

## Installation on Linux
OpenAI Gym and mujoco-py are outdated and the troubleshooting guides are mostly for Ubuntu, so the setup requires extra care. This guide uses `venv` to set up the environment instead of `conda`, and points to an older version of `GCC` to get the packages to compile. This guide is written mainly for Arch based distributions, you need to modify the installation commands for `Python3.8` and `gcc13`(or `gcc12`) to fit your linux distribution.

1. Install *Python3.8* (AUR)
```bash
paru -S python38
```
2. Create the virtual environment using *venv*
```bash
python3.8 -m venv mldl
```
3. Activate virtual environment
```bash
source mldl/bin/activate
```
4. Set up virtual environment to match the given [specification](https://github.com/lambdavi/rl_mldl_25/tree/main?tab=readme-ov-file#1-local-on-linux-recommended)
```bash
(mldl) pip install pip==22.0.1
(mldl) pip install setuptools==65.5.0 wheel==0.38
```
5. Get Mujoco
```bash
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv ./mujoco210-linux-x86_64.tar.gz ~/.mujoco
cd ~/.mujoco
tar -xf mujoco210-linux-x86_64.tar.gz
```
6. Clone this repository
```bash
git clone https://github.com/canyalniz/rl_mldl_25.git
```
7. Install requirements
```bash
(mldl) cd /path/to/rl_mldl_25
(mldl) pip install -r requirements.txt
```
8. Include the following lines in your .zshrc (or .bashrc)
```
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
9. Install GCC13
```bash
paru -S gcc13
```
10. Run *test_random_policy.py*
```bash
(mldl) export CC=gcc-13
(mldl) python test_random_policy.py
```

## Algorithms & Experiments
### REINFORCE
[Train](train.py) and [test](test.py) the vanilla policy gradient method REINFORCE with or without baselines. The model used for REINFORCE is [here](agent.py).
### PPO
#### Folder Structure
Before you run any PPO related scripts, you must first create a dedicated `logs & models` directory. By default this is assumed to be a folder named `logs_and_models` in the same directory as the scripts. This will be populated by dedicated folders for each run and will hold the produced model as well as logs regarding training and evaluation.

Due to the convention of StableBaselines3 Monitor logs we have one model per run for our use case.

For hyperparameter optimization the script expects there to be a dedicated folder called `optuna_studies` in the same directory as itself.

The default behaviors can be changed by passing commandline arguments to the scripts.
#### Hyperparamter Optimization
Use the dedicated [script](optuna_sb3.py) to optimize the PPO hyperparameters using Optuna. Assumes you have sqlite installed for persistent Optuna studies.
#### Train
Use the dedicated [script](train_sb3.py) to train your PPO model. The best model seen during the run is saved in the run directory, along with the training reward logs. Check the command line arguments for customization options.
#### Finetune
Use the dedicated [script](finetune_sb3.py) to fine-tune your PPO model. The finetuning script takes an existing model and continues training from where it was left off by first decreasing the learning rate and then following a linear learning rate decay schedule. Check the command line arguments for customization options.
#### Evaluate
Use the dedicated [script](evaluate_sb3.py) to evaluate your PPO model. The evaluation script takes an existing model and runs evaluation on it. Since we want to compare the performance of multiple models on different environments, the evaluation results of the models from the same run directory are collected into the same `eval_records.csv` file. Check the command line arguments for customization options.
#### Render
Use the dedicated [script](render_sb3.py) to render a run of your PPO model in the given Hopper environment. Check the command line arguments for customization options.

## Domain Randomization
### UDR
To train or evaluate a model using UDR on the Hopper environment, pass in "source-udr" as the dedicated command-line parameter. This picks the `CustomHopper-source-UDR-v0` environment defined in our [custom environments](env/custom_hopper.py).
### DROPO
Use the dedicated [script](run_dropo.py) to run the DROPO optimization algorithm for adaptive domain randomization. This will give you the mean vector and std vector for your parameters. You can then put these parameters in the dedicated variables in our [custom environments](env/custom_hopper.py).

To train or evaluate using the environment randomized according to the calculated DROPO parameters, pass in "source-normal" as the dedicated command-line parameter. This picks the `CustomHopper-source-normal-v0` environment defined in our [custom environments](env/custom_hopper.py).