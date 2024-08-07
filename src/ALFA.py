import GymWrapper as gw
import time
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common import on_policy_algorithm
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# Custom callback for logging during training


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_rollout_start(self):
        print("Rollout started")

    def _on_rollout_end(self):
        print("Rollout ended")

    def _on_step(self):
        return True


# Hyperparameters for training
BATCH_SIZE = 20  # Batch size for PPO (default is 64)
N_STEPS = SIM_TIME * 4  # Number of steps per environment (default is 2048)

# Meta-learning configuration
# train_num_scenarios = 20  # Number of scenarios for training
# test_num_scenarios = 5  # Number of scenarios for testing
# scenario_batch_size = 2  # Number of scenarios in each batch
# num_inner_updates = N_EPISODES  # Number of updates in the inner loop
# num_outer_updates = 2  # Number of meta-learning iterations
train_scenario_batch_size = 10  # Batch size for random chosen scenarios
test_scenario_batch_size = 5  # Batch size for random chosen scenarios
num_inner_updates = N_EPISODES  # Number of gradient steps for adaptation
num_outer_updates = 700  # Number of outer loop updates -> meta-training iterations


# Define the meta model architecture


class Meta_Model(nn.Module):
    def __init__(self, dim, lr=0.001):
        self.dim = dim
        super(Meta_Model, self).__init__()
        # Neural network layers
        self.fc1 = nn.Linear(self.dim, 32)
        self.fc2 = nn.Linear(32, 32)
        # Output layer for two values (alpha, beta)
        self.fc3 = nn.Linear(32, 2)
        # Optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Sigmoid activation to output values in [0, 1]
        output = torch.sigmoid(self.fc3(x))
        return output

    def update_model(self, query_losses):
        # Update the meta model based on the accumulated loss
        '''
        loss_len = len(Loop_loss)
        tensor_loss = torch.tensor(
            Loop_loss, dtype=torch.float64, requires_grad=True).cuda()
        tensor_zeros = torch.zeros(
            loss_len, dtype=torch.float64, requires_grad=True).cuda()
        loss = F.mse_loss(tensor_loss, tensor_zeros)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        '''
        # query_losses: 각 태스크의 쿼리 세트에 대한 손실 리스트
        meta_loss = sum(query_losses)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()


# Define the base model for reinforcement learning


class Base_Model:
    def __init__(self, env, policy="MlpPolicy"):
        self.env = env
        self.policy = policy
        self.inner_model = PPO(
            self.policy, self.env, verbose=0, n_steps=N_STEPS, batch_size=BATCH_SIZE)
        self.Loop_Loss = []
        self.writer = SummaryWriter(
            log_dir='./ALFA_tensorboard_logs')  # TensorBoard writer

    def cal_loss(self):
        # Calculate the loss for the PPO model
        clip_range = self.inner_model.clip_range(
            self.inner_model._current_progress_remaining)
        if self.inner_model.clip_range_vf is not None:
            clip_range_vf = self.inner_model.clip_range_vf(
                self.inner_model._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        for rollout_data in self.inner_model.rollout_buffer.get(self.inner_model.batch_size):
            actions = rollout_data.actions
            if isinstance(self.inner_model.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()
            if self.inner_model.use_sde:
                self.inner_model.policy.reset_noise(
                    self.inner_model.batch_size)

            values, log_prob, entropy = self.inner_model.policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()
            advantages = rollout_data.advantages
            if self.inner_model.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / \
                    (advantages.std() + 1e-8)

            ratio = torch.exp(log_prob - rollout_data.old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * \
                torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            if self.inner_model.clip_range_vf is None:
                values_pred = values
            else:
                values_pred = rollout_data.old_values + \
                    torch.clamp(values - rollout_data.old_values, -
                                clip_range_vf, clip_range_vf)

            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            entropy_loss = - \
                torch.mean(-log_prob) if entropy is None else - \
                torch.mean(entropy)
            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + self.inner_model.ent_coef * \
                entropy_loss + self.inner_model.vf_coef * value_loss

        return loss

    def custom_train(self, meta_model):
        # Train the base model with updates from the meta model
        self.inner_model.policy.set_training_mode(True)
        clip_range = self.inner_model.clip_range(
            self.inner_model._current_progress_remaining)
        if self.inner_model.clip_range_vf is not None:
            clip_range_vf = self.inner_model.clip_range_vf(
                self.inner_model._current_progress_remaining)

        for rollout_data in self.inner_model.rollout_buffer.get(self.inner_model.batch_size):
            loss = self.cal_loss()
            current_params = {k: v.clone().cuda()
                              for k, v in self.inner_model.policy.state_dict().items()}
            self.inner_model.policy.optimizer.zero_grad()
            loss.backward()
            self.inner_model.policy.optimizer.step()

            grad_loss = {k: v.grad.clone().cuda()
                         for k, v in self.inner_model.policy.named_parameters()}
            current_params_list = list(current_params.values())
            grad_loss_list = list(grad_loss.values())
            current_params_flat = torch.cat(
                [p.view(-1) for p in current_params_list]).cuda()
            grad_loss_flat = torch.cat([g.view(-1)
                                       for g in grad_loss_list]).cuda()
            input_data = torch.cat(
                (current_params_flat, grad_loss_flat)).unsqueeze(0).cuda()

            result_lst = meta_model(input_data)
            alpha, beta = result_lst[0][0], result_lst[0][1]
            with torch.no_grad():
                for key in current_params.keys():
                    current_params[key] = beta * \
                        current_params[key] - alpha * grad_loss[key]

            for name, param in self.inner_model.policy.named_parameters():
                param.grad = torch.clamp(
                    param.grad, -self.inner_model.max_grad_norm, self.inner_model.max_grad_norm)
            self.inner_model.policy.optimizer.step()
            self.Loop_Loss.append(self.cal_loss())

    def update_base(self, scenario, meta_model, num_inner_updates):
        # Update the base model for a given scenario
        self.inner_model = PPO(
            self.policy, self.env, verbose=0, n_steps=N_STEPS, batch_size=BATCH_SIZE)
        buffer = self.inner_model.rollout_buffer
        for eps in range(num_inner_updates):
            self.inner_model.rollout_buffer.reset()
            for x in range(N_STEPS // SIM_TIME):
                obs = self.env.reset()
                for day in range(SIM_TIME):
                    obs = torch.tensor([obs], dtype=torch.float64).cuda()
                    action, value, log_prob = self.inner_model.policy.forward(
                        obs)
                    numpy_array = action[0].cpu().numpy()
                    list_action = numpy_array.tolist()
                    new_obs, reward, done, info = self.env.step(list_action)
                    done = 1 if done else 0
                    buffer.add(obs.cpu(), action.cpu(), reward,
                               done, value.detach(), log_prob.detach())
                    obs = new_obs
            self.custom_train(meta_model)
        print(gw.evaluate_model(self.inner_model, self.env, SIM_TIME))

    def log_to_tensorboard(self, iteration):
        # Log the loss to TensorBoard
        self.writer.add_scalar("Total_Loss", sum(self.Loop_Loss), iteration)


# Main function to start the training process


def main():

    env = GymInterface()
    base_model = Base_Model(env)
    base_model.inner_model.learn(SIM_TIME * 4)

    current_params = {k: v.clone().cuda()
                      for k, v in base_model.inner_model.policy.state_dict().items()}
    current_params_list = list(current_params.values())
    current_params_flat = torch.cat([p.view(-1)
                                    for p in current_params_list]).cuda()

    meta_model = Meta_Model(len(current_params_flat) * 2).cuda()

    for iteration in range(num_outer_updates):
        # Sample a batch of scenarios
        scenario_batch = [Create_scenario(DIST_TYPE)
                          for _ in range(train_scenario_batch_size)]
        query_losses = []
        for scenario in scenario_batch:
            print("\n\nTRAINING SCENARIO: ", scenario)
            print("\nOuter Loop: ", env.cur_outer_loop,
                  " / Inner Loop: ", env.cur_inner_loop)

            # Reset the scenario for the environment
            print("Scenario: ",  scenario)

            base_model.update_base(scenario, meta_model, num_inner_updates)

            query_loss = base_model.cal_loss()  # 이 메서드를 수정하여 쿼리 세트에 대한 손실을 반환하도록 해야 함
            query_losses.append(query_loss)

        base_model.log_to_tensorboard(iteration)
        meta_model.update_model(base_model.Loop_Loss)
        # torch.save({'model_state_dict': meta_model.state_dict(),
        #             'optimizer_state_dict': meta_model.optimizer.state_dict()},
        #            'model_checkpoint.pth')


start_time = time.time()
main()
# Calculate computation time and print it
end_time = time.time()

print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n")
