import GymWrapper as gw
import time
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from gym import spaces

from torch.utils.tensorboard import SummaryWriter


K = N_EPISODES
# Hyperparameters
ALPHA = 0.0001  # Inner loop step size (사용되지 않는 값) ->  SB3 PPO 기본 값(0.0003)
BATCH_SIZE = 20  # Default 64
N_STEPS = SIM_TIME*K  # Default 2048

BETA = 0.0003  # Outer loop step size ## Default: 0.001
train_scenario_batch_size = 10  # Batch size for random chosen scenarios
test_scenario_batch_size = 5  # Batch size for random chosen scenarios
num_outer_updates = 500  # Number of outer loop updates -> meta-training iterations

# Meta-learning algorithm


class MetaLearner:
    def __init__(self, env, policy='MlpPolicy', alpha=ALPHA, beta=BETA):
        """
        Initializes the MetaLearner with the specified environment and hyperparameters.
        """
        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.beta = beta

        self.meta_model = PPO(policy, self.env, verbose=0,
                              n_steps=N_STEPS, learning_rate=self.beta, batch_size=BATCH_SIZE, n_epochs=1)
        self.writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)

    def inner_loop(self, K=K):
        """
        Adapts the meta-policy to a specific task using gradient descent.
        """
        self.env.reset()
        # n_steps: K개의 rollout을 가지고 policy를 업데이트
        adapted_model = PPO(self.policy, self.env, verbose=0,
                            n_steps=N_STEPS, learning_rate=self.alpha, batch_size=BATCH_SIZE, n_epochs=1)

        # 정책 네트워크의 파라미터 복사
        adapted_model.policy.load_state_dict(
            self.meta_model.policy.state_dict())

        # LINE 5 - 7: 정책 네트워크의 파라미터의 파라미터 업데이트
        # (SIM_TIME*K) timestep 길이의 rollout을 두 번 학습하고, 두 번째 학습에 사용된 rollout (D')을 buffer에 저장함
        adapted_model.learn(total_timesteps=N_STEPS*2)

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
        # # Optional: clip range for the value function
        # if self.meta_model.clip_range_vf is not None:
        #     clip_range_vf = self.meta_model.clip_range_vf(
        #         self.meta_model._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # Do a complete pass on the rollout buffer
        for rollout_data in self.meta_model.rollout_buffer.get(self.meta_model.batch_size):
            actions = rollout_data.actions
            if isinstance(self.meta_model.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.meta_model.use_sde:
                self.meta_model.policy.reset_noise(
                    self.meta_model.batch_size)

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

            # Logging
            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            values_pred = values
            # if self.meta_model.clip_range_vf is None:
            #     # No clipping
            #     values_pred = values
            # else:
            #     # Clip torche difference between old and new value
            #     # NOTE: torchis depends on torche reward scaling
            #     values_pred = rollout_data.old_values + torch.clamp(
            #         values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            #     )

            # Value loss using torche TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + self.meta_model.ent_coef * \
                entropy_loss + self.meta_model.vf_coef * value_loss

            # Optimization step
            self.meta_model.policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(
                self.meta_model.policy.parameters(), self.meta_model.max_grad_norm)
            self.meta_model.policy.optimizer.step()

    def meta_update(self, rollout_list):
        for rollout_buffer in rollout_list:
            self.meta_model.rollout_buffer = rollout_buffer
            self.custom_train()

    def meta_test(self):
        """
        Performs the meta-test step by averaging gradients across scenarios.
        """
        # eval_scenario = Create_scenario(DIST_TYPE)
        test_scenario_batch = [Create_scenario()
                               for _ in range(test_scenario_batch_size)]

        # Set the scenario for the environment
        all_rewards = []
        for test_scenario in test_scenario_batch:
            # for test_scenario in test_scenario_batch:
            self.env.reset()
            self.env.scenario = test_scenario
            print("\n\nTEST SCENARIO: ", self.env.scenario)
            meta_mean_reward, meta_std_reward = gw.evaluate_model(
                self.meta_model, self.env, N_EVAL_EPISODES)
            all_rewards.append(meta_mean_reward)

        # Calculate mean reward across all episodes
        meta_mean_reward = np.mean(all_rewards)
        self.log_to_tensorboard(iteration, meta_mean_reward, meta_std_reward)

        return meta_mean_reward, meta_std_reward

    def log_to_tensorboard(self, iteration, mean_reward, std_reward):
        """
        Logs the metrics to TensorBoard.
        """
        self.writer.add_scalar("Reward/Mean", mean_reward, iteration)
        self.writer.add_scalar("Reward/Std", std_reward, iteration)


# Start timing the computation
start_time = time.time()

# Create environment
env = GymInterface()

# Training the Meta-Learner
meta_learner = MetaLearner(env)

meta_rewards = []
random_rewards = []

for iteration in range(num_outer_updates):
    env.scenario_batch_size = train_scenario_batch_size
    # LINE 3: Sample a batch of scenarios
    scenario_batch = [Create_scenario()
                      for _ in range(train_scenario_batch_size)]
    if iteration == num_outer_updates-1:
        meta_learner.env.outer_end = True
    # Adapt the meta-policy to each scenario in the batch
    rollout_list = []
    for scenario in scenario_batch:  # LINE 4
        print("\n\nTRAINING SCENARIO(DEMAND): ", scenario["DEMAND"])
        print("\n\nTRAINING SCENARIO(LEAD_TIME): ", scenario["LEADTIME"])
        print("\nOuter Loop: ", env.cur_outer_loop,
              " / Inner Loop: ", env.cur_inner_loop)

        # Reset the scenario for the environment
        meta_learner.env.scenario = scenario
        print("Demand: ", meta_learner.env.scenario["DEMAND"])
        print("Lead_time: ", meta_learner.env.scenario["LEADTIME"])
        # LINE 5 - 7
        adapted_model = meta_learner.inner_loop()  # LINE 6-7

        # LINE 8: 학습된 모델로부터 rollout 수집
        # rollout buffer에는 K개의 에피소드가 저장되어 있음
        meta_learner.meta_model.rollout_buffer = adapted_model.rollout_buffer
        meta_learner.custom_train()
        # rollout_list.append(rollout_buffer)
        '''
        # LINE 8: 학습된 모델로부터 rollout 수집
        rollout_buffer = adapted_model.rollout_buffer
        rollout_list.append(rollout_buffer)
        '''

        env.cur_episode = 1
        env.cur_inner_loop += 1

    # # LINE 10: Perform the meta-update step
    # meta_learner.meta_update(rollout_list)

    # Evaluate the meta-policy on the test scenario
    meta_mean_reward, meta_std_reward = meta_learner.meta_test()
    meta_rewards.append(meta_mean_reward)
    print(
        f'Iteration {iteration+1}/{num_outer_updates} - Mean Reward: {meta_mean_reward:.2f} ± {meta_std_reward:.2f}\n')
    print('===========================================================')

    env.cur_episode = 1
    env.cur_inner_loop = 1
    env.cur_outer_loop += 1
    env.outer_end = False

    # Save the trained meta-policy
    meta_learner.meta_model.save(SAVED_MODEL_NAME)

training_end_time = time.time()

print("\nMETA TRAINING COMPLETE \n\n\n")

# Calculate computation time and print it
end_time = time.time()


# Evaluate the trained meta-policy
meta_mean_reward, meta_std_reward = meta_learner.meta_test()
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {meta_mean_reward:.2f} +/- {meta_std_reward:.2f}")

print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
      f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
      f"Test time:{(end_time - training_end_time)/60:.2f} minutes")

# Optionally render the environment
env.render()
