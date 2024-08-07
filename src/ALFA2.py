import GymWrapper as gw
import time
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from gym import spaces

from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
BATCH_SIZE = 20
N_STEPS = SIM_TIME*4
BETA = 0.0003
train_scenario_batch_size = 10
test_scenario_batch_size = 5
num_inner_updates = N_EPISODES
num_outer_updates = 700

# Meta network for generating hyperparameters


class Meta_Model(nn.Module):
    def __init__(self, input_dim):
        super(Meta_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)  # 1 output: alpha
        self.optimizer = torch.optim.Adam(self.parameters(), lr=BETA)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Ensure output is between 0 and 1

# Meta-learning algorithm


class MetaLearner:
    def __init__(self, env, policy='MlpPolicy', beta=BETA):
        self.env = env
        self.policy = policy
        self.beta = beta

        self.meta_model = PPO(policy, self.env, verbose=0,
                              n_steps=N_STEPS, learning_rate=self.alpha, batch_size=BATCH_SIZE)

        # Initialize meta network
        # params + grads
        input_dim = sum(p.numel()
                        for p in self.meta_model.policy.parameters()) * 2
        self.meta_network = Meta_Model(input_dim)

        self.writer = SummaryWriter(log_dir='./META_tensorboard_logs')

    def inner_loop(self, num_updates=num_inner_updates):
        self.env.reset()
        adapted_model = PPO(self.policy, self.env, verbose=0,
                            n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=1, learning_rate=alpha)
        adapted_model.policy.load_state_dict(
            self.meta_model.policy.state_dict())

        for _ in range(10):  # 특정 시나리오에 대해 반복하는것 (S)

            # Generate adaptive learning rate
            params = torch.cat([p.view(-1)
                               for p in adapted_model.policy.parameters()])
            grads = torch.cat(
                [p.grad.view(-1) for p in adapted_model.policy.parameters() if p.grad is not None])
            input_data = torch.cat((params, grads))

            alpha = self.meta_network(input_data)

            # # Apply adaptive update
            # with torch.no_grad():
            #     for param in adapted_model.policy.parameters():
            #         if param.grad is not None:
            #             param.data -= alpha * param.grad

            adapted_model.learn(total_timesteps=SIM_TIME)

            # 학습된 모델로부터 rollout 수집
            rollout_buffer = adapted_model.rollout_buffer
            self.update_meta_parameter(rollout_buffer)

        return adapted_model

    def custom_train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.meta_model.policy.set_training_mode(True)
        # Compute current clip range
        clip_range = self.meta_model.clip_range(
            self.meta_model._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.meta_model.clip_range_vf is not None:
            clip_range_vf = self.meta_model.clip_range_vf(
                self.meta_model._current_progress_remaining)  # type: ignore[operator]

        # Do a complete pass on the rollout buffer
        for rollout_data in self.meta_model.rollout_buffer.get(self.meta_model.batch_size):
            actions = rollout_data.actions
            if isinstance(self.meta_model.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # # Re-sample the noise matrix because the log_std has changed
            # if self.meta_model.use_sde:
            #     self.meta_model.policy.reset_noise(
            #         self.meta_model.batch_size)

            values, log_prob, entropy = self.meta_model.policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.meta_model.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()
                              ) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * \
                torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            if self.meta_model.clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip torche difference between old and new value
                # NOTE: torchis depends on torche reward scaling
                values_pred = rollout_data.old_values + torch.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                )
            # Value loss using torche TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = policy_loss + self.meta_model.ent_coef * \
                entropy_loss + self.meta_model.vf_coef * value_loss

            # Optimization step
            self.meta_model.policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(
                self.meta_model.policy.parameters(), self.meta_model.max_grad_norm)

            self.meta_model.policy.optimizer.step()

            # # Generate adaptive hyperparameters
            # params = torch.cat([p.view(-1) for p in model.policy.parameters()])
            # grads = torch.cat(
            #     [p.grad.view(-1) for p in model.policy.parameters() if p.grad is not None])
            # input_data = torch.cat((params, grads))
            # alpha, beta = self.meta_network(input_data)

            # # Apply adaptive update
            # with torch.no_grad():
            #     for param in model.policy.parameters():
            #         if param.grad is not None:
            #             param.data = beta * param.data - alpha * param.grad

        return loss

    def update_meta_parameter(self, rollout_buffer):
        self.meta_model.rollout_buffer = rollout_buffer
        self.custom_train()

    def meta_update(self, losses):
        meta_loss = sum(losses)
        self.meta_network.optimizer.zero_grad()
        meta_loss.backward()
        self.meta_network.optimizer.step()

    def meta_test(self):
        test_scenario_batch = [Create_scenario(
            DIST_TYPE) for _ in range(test_scenario_batch_size)]
        all_rewards = []
        for test_scenario in test_scenario_batch:
            self.env.reset()
            self.env.scenario = test_scenario
            print("\n\nTEST SCENARIO: ", self.env.scenario)
            meta_mean_reward, meta_std_reward = gw.evaluate_model(
                self.meta_model, self.env, N_EVAL_EPISODES)
            all_rewards.append(meta_mean_reward)

        meta_mean_reward = np.mean(all_rewards)
        self.log_to_tensorboard(iteration, meta_mean_reward, meta_std_reward)
        return meta_mean_reward, meta_std_reward

    def log_to_tensorboard(self, iteration, mean_reward, std_reward):
        self.writer.add_scalar("Reward/Mean", mean_reward, iteration)
        self.writer.add_scalar("Reward/Std", std_reward, iteration)


# Main training loop
start_time = time.time()
env = GymInterface()
meta_learner = MetaLearner(env)

for iteration in range(num_outer_updates):
    scenario_batch = [Create_scenario(DIST_TYPE)
                      for _ in range(train_scenario_batch_size)]
    losses = []

    for scenario in scenario_batch:
        print("\n\nTRAINING SCENARIO: ", scenario)
        print("\nOuter Loop: ", env.cur_outer_loop,
              " / Inner Loop: ", env.cur_inner_loop)
        meta_learner.env.scenario = scenario
        adapted_model = meta_learner.inner_loop()
        loss = meta_learner.custom_train(adapted_model)
        losses.append(loss)
        env.cur_episode = 1
        env.cur_inner_loop += 1

    meta_learner.meta_update(losses)
    meta_mean_reward, meta_std_reward = meta_learner.meta_test()
    print(
        f'Iteration {iteration+1}/{num_outer_updates} - Mean Reward: {meta_mean_reward:.2f} ± {meta_std_reward:.2f}\n')
    print('===========================================================')

    env.cur_episode = 1
    env.cur_inner_loop = 1
    env.cur_outer_loop += 1

    meta_learner.meta_model.save(SAVED_MODEL_NAME)

training_end_time = time.time()
print("\nMETA TRAINING COMPLETE \n\n\n")

# Final evaluation and timing
meta_mean_reward, meta_std_reward = meta_learner.meta_test()
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {meta_mean_reward:.2f} +/- {meta_std_reward:.2f}")

end_time = time.time()
print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
      f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
      f"Test time:{(end_time - training_end_time)/60:.2f} minutes")

env.render()
