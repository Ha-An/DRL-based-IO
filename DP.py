import numpy as np
import pprint
from config import *


# 임의로 상태 정의
state_space = []
for i in range(10):
    for j in range(10):
            state_space.append([i, j])
state_size = len(state_space)
# 상태 및 행동 공간 확인
# print(state_space)
# print(action_space)


# 상태 전이 확률, 보상, 감쇠인자 정의
tran_prob = 1
# reward = - total_cost_per_day[-1]
reward = -1  # 재설정 필요
gamma = discount_factor

# 초기 정책 설정
policy = {}
for state in state_space:
    prob_per_action = 1.0 / len(action_space)  # 가능한 모든 행동에 대해 같은 확률 부여
    # policy[tuple(state)] = {tuple(action): prob_per_action for action in action_space}  # 확률 부여
    policy[tuple(state)] = {action: prob_per_action for action in action_space}
# 초기 정책 설정 확인
# print(policy[tuple([0, 0])])

# 초기 가치 함수 초기화 (예: 모든 상태에 대해 0으로 초기화)
value_function = np.zeros([10,10])
# 초기 가치 함수 초기화 확인
# print(value_function)

# 반복 횟수
num_iterations = 100



# # 정책 반복 (다음 상태 정의를 안했음)
# def policy_iteration():
#     global value_function, policy
#     for _ in range(num_iterations):
#         new_value_function = np.zeros_like(value_function)
#         new_policy = {}

#         for state in state_space:
#             expected_values = []

#             # 정책 평가
#             for action in action_space:
#                 expected_value = 0

#                 for next_state in state_space:
#                     next_value = value_function[next_state[0], next_state[1]]
#                     expected_value += policy[tuple(state)][tuple(action)] * (reward + gamma * tran_prob * next_value)
#                 expected_values.append(expected_value)

#             # 정책 개선
#             best_action_index = np.argmax(expected_values)
#             best_action = action_space[best_action_index]
#             new_policy[tuple(state)] = {tuple(action): 1.0 if tuple(action) == best_action else 0.0 for action in action_space}
            
#             # 새로운 가치 함수로 업데이트
#             new_value_function[state[0], state[1]] = max(expected_values)
        
#         value_function = np.round(new_value_function,3)
#         policy = new_policy
             
#     # 최종적으로 계산된 가치 함수와 정책 출력
#     print("Value Function : ", value_function)
#     print("Policy : ", policy)

# policy_iteration()


# # 가치 반복 (다음 상태 정의를 안했음)
# def value_iteration():
#     global value_function
#     for _ in range(num_iterations):
#         new_value_function = np.zeros_like(value_function)

#         for state in state_space:
#             max_value = -np.inf

#             # 정책 평가
#             for action in action_space:
#                 action_value = 0

#                 for next_state in state_space:
#                     next_value = value_function[next_state[0],next_state[1]]
#                     action_value += (reward + gamma * tran_prob * next_value)
#                 max_value = max(max_value, action_value)
                
#             # 새로운 가치 함수로 업데이트
#             new_value_function[state[0], state[1]] = max_value

#         value_function = np.round(new_value_function,3)

#     # 최종적으로 계산된 가치 함수 출력
#     print("Value Function : ", value_function)

# value_iteration()



# s'(다음 상태) 정의
def calculate_next_state(state, action):
    s1, s2 = state
    # (s_prime1, s_prime2) = (s1 + P[0]['PRODUCTION_RATE'] - I[0]['DEMAND_QUANTITY'], s2 - P[0]['PRODUCTION_RATE'] + action)
    s_prime1 = max(0, min(9, s1 + P[0]['PRODUCTION_RATE'] - I[0]['DEMAND_QUANTITY']))  # +(생산량), -(고객 주문량)
    s_prime2 = max(0, min(9, s2 - P[0]['PRODUCTION_RATE'] + action))  #  -(생산량), +(내 주문량)

    return (s_prime1, s_prime2)
# 다음 상태 확인
# print(calculate_next_state((5,5),2))



# # 정책 반복
# def policy_iteration():
#     global value_function, policy
#     for _ in range(num_iterations):
#         new_value_function = np.zeros_like(value_function)
#         new_policy = {}

#         for state in state_space:
#             expected_values = []

#             # 정책 평가
#             for action in action_space:
#                 expected_value = 0
#                 s_prime1, s_prime2 = calculate_next_state(state, action)
#                 reward = -(s_prime1 * I[0]['HOLD_COST'] + s_prime2 * I[1]['HOLD_COST'])  # Holding cost를 곱해줌 (Product : 5, Raw Material : 1)
#                 next_value = value_function[s_prime1, s_prime2]
#                 expected_value += policy[tuple(state)][action] * (reward + gamma * tran_prob * next_value)  # 벨만 기대 방정식
#                 expected_values.append(expected_value)

#             # 정책 개선
#             best_action_index = np.argmax(expected_values)
#             best_action = action_space[best_action_index]
#             new_policy[tuple(state)] = {action: 1.0 if action == best_action else 0.0 for action in action_space}
            
#             # 새로운 가치 함수로 업데이트
#             new_value_function[state[0], state[1]] = max(expected_values)
        
#         value_function = np.round(new_value_function,3)
#         policy = new_policy
             
#     # 최종적으로 계산된 가치 함수와 정책 출력
#     # print("Value Function : ", value_function)
#     # print("Policy : ", policy)
#     print("Value Function : ")
#     pprint.pprint(value_function)
#     print("Policy : ")
#     pprint.pprint(policy)

# policy_iteration()


# 가치 반복
def value_iteration():
    global value_function
    for _ in range(num_iterations):
        new_value_function = np.zeros_like(value_function)

        for state in state_space:
            max_value = -np.inf

            # 정책 평가
            for action in action_space:
                action_value = 0
                s_prime1, s_prime2 = calculate_next_state(state, action)
                reward = -(s_prime1 * I[0]['HOLD_COST'] + s_prime2 * I[1]['HOLD_COST'])
                next_value = value_function[s_prime1, s_prime2]
                action_value += (reward + gamma * tran_prob * next_value)  # 벨만 최적 방정식
                max_value = max(max_value, action_value)
                
            # 새로운 가치 함수로 업데이트
            new_value_function[state[0], state[1]] = max_value

        value_function = np.round(new_value_function,3)

    # 최종적으로 계산된 가치 함수 출력
    print("Value Function : ")
    pprint.pprint(value_function)

value_iteration()