from InventoryMgtEnv import GymInterface
from config import *
import numpy as np
import HyperparamTuning as ht  # 하이퍼파라미터 튜닝 기능 제공
import time
from stable_baselines3 import DQN, DDPG, PPO

# Create environment

env = GymInterface() # 강화학습에 사용될 환경을 생성  inventorymgtenv go 


def evaluate_model(model, env, num_episodes):
    all_rewards = []
    for _ in range(num_episodes): #_는 에피소드  
        obs = env.reset() # 환경을 초기상태로 리셋 > 에피소드마다 수행  obs는 관찰
        episode_reward = 0 # 리워드를 0으로 리렛
        done = False
        while not done:
            action, _ = model.predict(obs)  # predict는 적절한 액션을 예측
            obs, reward, done, _ = env.step(action) # 예측된 액션을 환경에 적용함 obs,reward done 추가정보를 받음
            episode_reward += reward  
        all_rewards.append(episode_reward)  
    mean_reward = np.mean(all_rewards)  #보상의 평균
    std_reward = np.std(all_rewards) # 보상의 표준편차 
    return mean_reward, std_reward


def build_model():  # 강화학습 모델을 구축하는 역할
    if RL_ALGORITHM == "DQN":
        # model = DQN("MlpPolicy", env, verbose=0)
        model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                    batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                     batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                    batch_size=BEST_PARAMS['batch_size'], verbose=0)
        print(env.observation_space)
    return model

#MlpPolicy = 멀티레이어퍼셉트론 
#learning_rate = 모델이 학습하는 속도를 결정 


start_time = time.time()

if OPTIMIZE_HYPERPARAMETERS: #config에 파일있음
    ht.run_optuna(env)

model = build_model()
model.learn(total_timesteps=SIM_TIME*N_EPISODES)  # Time steps = days
env.render() 

# 학습 후 모델 평가
mean_reward, std_reward = evaluate_model(model, env, N_EVAL_EPISODES)
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

# # Optimal policy
# if RL_ALGORITHM == "DQN":
#     optimal_actions_matrix = np.zeros(
#         (INVEN_LEVEL_MAX + 1, INVEN_LEVEL_MAX + 1), dtype=int)
#     for i in range(INVEN_LEVEL_MAX + 1):
#         for j in range(INVEN_LEVEL_MAX + 1):
#             if STATE_DEMAND:
#                 state = np.array([i, j, I[0]['DEMAND_QUANTITY']])
#                 action, _ = model.predict(state)
#                 optimal_actions_matrix[i, j] = action
#             else:
#                 state = np.array([i, j])
#                 action, _ = model.predict(state)
#                 optimal_actions_matrix[i, j] = action

#     # Print the optimal actions matrix
#     print("Optimal Actions Matrix:")
#     # print("Demand quantity: ", I[0]['DEMAND_QUANTITY'])
#     print(optimal_actions_matrix)

end_time = time.time()
print(f"Computation time: {(end_time - start_time)/3600:.2f} hours")

'''
#모델 저장 및 로드 (선택적)
model.save("dqn_inventory")
loaded_model = DQN.load("dqn_inventory")
'''

# TensorBoard 실행:
# tensorboard --logdir="C:/tensorboard_logs/"
# http://localhost:6006/
