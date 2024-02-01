"""
Train the demonstration policy.
Refer to paper Appendix B.2 or B.4.
"""

import safety_gymnasium as gym
from env_wrapper import SafetyGoalFeatureWrapper
from stable_baselines3.ppo.var_ppo import VaRPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import time
import torch.nn as nn
import os

def create_env(env_name, log_path):
    env = gym.make(env_name)
    env = SafetyGoalFeatureWrapper(env, env_name, mode="true")
    env = Monitor(env, log_path)
    return env

def candidate_selection(args, verbose=True):

    start_time = time.time()

    # checkpoint/monitor path
    log_path = os.path.join("./candidate_selection", args.env_name, "true")
    os.makedirs(log_path, exist_ok=True)
    Monitor.EXT = f"monitor_seed{args.seed}.csv"

    # create env
    env = create_env(args.env_name, log_path)
    n_rewards = env.get_n_rewards()

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq = args.save_freq,
        save_path = log_path,
        name_prefix = f"rl_model_seed{args.seed}",
        save_replay_buffer=True
    )

    policy_kwargs = dict(activation_fn=nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = VaRPPO("MlpPolicy", env, n_rewards, verbose=1,
        policy_kwargs=policy_kwargs, learning_rate=1e-4, n_steps=4000, seed=args.seed)

    print(model.policy)

    model.learn(args.total_timesteps, callback=checkpoint_callback)

    print("Time:", time.time() - start_time)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", help="Environment name. Goal: 'SafetyPointGoal5-v0', Circle: 'SafetyPointCircle1-v0'")
    parser.add_argument("--total_timesteps", default=10000, type=int, help="Total number of timesteps to train")
    parser.add_argument("--save_freq", default=5000, type=int, help="Frequency of saving model checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")

    args = parser.parse_args()

    # array batch jobs


    candidate_selection(args)
