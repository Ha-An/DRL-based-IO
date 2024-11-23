# DRL-based-IO
Deep Reinforcement Learning-based Inventory Optimization

<!-- # How to run DRL
* gym
* pandas
* torch
* tensorboard
* optuna
* stable_baselines3
* shimmmy>=0.2.1 -->

#  How to set parameters
All DRL parameters are stored in config_RL.py.
* DRL: To proceed with DRL, it must be set to True.
* N_EPISODES: Variable that sets the total number of learning steps.
* N_EVAL_EPISODES: Number of times to evaluate after learning.

Every parmeters in config_SimPy.py
* SIM_TIME: Set the period to simulate (days per episode)
* DEMAND_SCENARIO: Set distribution for customers to order
* LEADTIME_SCENARIO: Set the leadtime distribution
* USE_SQPOLICY: Must be set to Falsee. When using SQpolicy (DRL is NOT used)
* SQPAIR: Set when and how many raw materials to order. (Ordering rules : Reorder point (S) and Order quantity (Q))

# Description

## environment.py
* The code is a simulation environment for reinforcement learning.
* The code is from SimPy_IMS(https://github.com/Ha-An/SimPy_IMS; Version 1.0)
  <!-- * Remove line 136 of environment (Change Order input)
  * Unit processing cost modification (Processing cost->Processing cost/Processing time)
  * +line268: Add Delivery cost 
  * +line342~345: Shortage cost update pass
  * +line422~425: Add expected_shortage to state -->

# Contact
* Yosep Oh (yosepoh@hanyang.ac.kr)