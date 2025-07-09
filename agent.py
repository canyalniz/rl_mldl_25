import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class ValueEstimator(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.state_space = state_space
        self.hidden = 64
        self.activation_fn = torch.nn.Tanh()

        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, 1)


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        output = self.fc3(x)
        
        return output

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.activation_fn = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.activation_fn(self.fc1_actor(x))
        x_actor = self.activation_fn(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        
        return normal_dist


class Agent(object):
    def __init__(self, policy, run_id, value_function=None, critic=False, device='cpu', skip_over=0, check_freq=1000, model_name="best_model", verbose=1):
        self.train_device = device
        self.run_id = run_id
        self.verbose = verbose
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        if value_function:
            self.value_function = value_function.to(self.train_device)
            self.value_function_optimizer = torch.optim.Adam(value_function.parameters(), lr=1e-3)
        else:
            if critic:
                raise ValueError("Actor Critic agent needs to be initialized with a ValueEstimator.")

            self.value_function = None

        self.critic = critic
        self.I = 1 if critic else None

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.state = None
        self.next_state = None
        self.action_log_prob = None
        self.reward = None

        self.n_calls = 0
        self.last100rewards = np.zeros(100)
        self.best_mean_reward = -np.inf
        self.skip_over = skip_over
        self.check_freq = check_freq

        self.logs_dir = "logs_and_models"
        self.model_name = model_name

    def update_policy(self):
        if not self.critic:
            action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
            states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
            next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
            rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
            done = torch.Tensor(self.done).to(self.train_device)
            done = done[-1]

            self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

            #
            # TASK 2:
            #   - compute discounted returns
            discounted_returns = discount_rewards(rewards, self.gamma)

            #   - compute policy gradient loss function given actions and returns
            T = discounted_returns.shape[-1]
            discount_vector = torch.tensor([self.gamma ** t for t in range(T)], device=self.train_device)
            if self.value_function:
                baseline = self.value_function(states)
            else:
                baseline = 20
            
            delta = (discounted_returns - baseline).detach()
            policy_gradient_loss = -1 * torch.sum(discount_vector * delta * action_log_probs)
            
            if self.value_function:
                value_estimator_loss = -1 * torch.sum(delta * baseline)

            #   - compute gradients and step the optimizer
            #
            self.optimizer.zero_grad()
            policy_gradient_loss.backward()
            self.optimizer.step()

            if self.value_function:
                self.value_function_optimizer.zero_grad()
                value_estimator_loss.backward()
                self.value_function_optimizer.step()
        else:            
            past_state_value = self.value_function(self.state)
            next_state_value = 0 if self.done else self.value_function(self.next_state)

            
            delta = (self.reward + self.gamma * next_state_value - past_state_value).detach()
            
            value_estimator_loss = -1 * delta * past_state_value

            policy_gradient_loss = -1 * self.I * delta * self.action_log_prob

            self.I = self.I * self.gamma

            self.optimizer.zero_grad()
            policy_gradient_loss.backward()
            self.optimizer.step()

            self.value_function_optimizer.zero_grad()
            value_estimator_loss.backward()
            self.value_function_optimizer.step()

            if self.done:
                self.I = 1

        if (self.n_calls>=self.skip_over) and (self.n_calls % self.check_freq == 0):
            # Mean training reward over the last 100 episodes
            mean_reward = np.mean(self.last100rewards)

            print(f"Num updates: {self.n_calls}")
            print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                self.save_model()

        self.n_calls += 1

        return
    
    def save_model(self):
        save_path = os.path.join(self.logs_dir, self.run_id, self.model_name)
        if self.verbose >= 1:
            print(f"Saving new best model to {save_path}")
        torch.save(self.policy.state_dict(), save_path + "_policy.mdl")
        if not (self.value_function is None):
            torch.save(self.value_function.state_dict(), save_path + "_value_function.mdl")

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.last100rewards[self.n_calls % 100] = reward

        if not self.critic:
            self.states.append(torch.from_numpy(state).float())
            self.next_states.append(torch.from_numpy(next_state).float())
            self.action_log_probs.append(action_log_prob.unsqueeze(0))
            self.rewards.append(torch.Tensor([reward]))
            self.done.append(done)
        else:
            self.state = torch.from_numpy(state).float().to(self.train_device)
            self.next_state = torch.from_numpy(next_state).float().to(self.train_device)
            self.action_log_prob = action_log_prob.to(self.train_device)
            self.reward = reward
            self.done = done