import safety_gymnasium as gym
from env_wrapper import SafetyGoalFeatureWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.var_ppo import VaRPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import time
import torch.nn as nn
import os

os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

def create_env(env_name, mode, log_path, seed):
    env = gym.make(env_name)
    if mode == "trex":
        samples_filename = f"trex_reward_seed{seed}.pkl"
    else:    
        samples_filename = os.path.join(f"samples_seed{seed}.pkl")
    
    env = SafetyGoalFeatureWrapper(env, env_name, samples_filename=samples_filename, mode=mode)
    env = Monitor(env, log_path)

    return env

def candidate_selection(args, verbose=True):

    start_time = time.time()

    # checkpoint/monitor path
    log_path = os.path.join("./candidate_selection", args.env_name, args.mode, args.exp_name)
    os.makedirs(log_path, exist_ok=True)
    Monitor.EXT = f"monitor_seed{args.trial_seed}.csv"

    # create env
    env = create_env(args.env_name, args.mode, log_path, args.trial_seed)
    n_rewards = env.get_n_rewards()

    # create model
    policy_kwargs = dict(activation_fn=nn.ReLU,
            net_arch=dict(pi=[], vf=[]))

    model = VaRPPO("CnnPolicy", env, n_rewards, alpha=args.alpha, verbose=1,
        policy_kwargs=policy_kwargs, learning_rate=args.lr, seed=args.seed)

    if args.mode == "dist":
        # POSTPI
        # set expected return of initial policy under reward samples
        # so that algorithm improves over initial policy
        import pickle
        with open(os.path.join("evaluation", args.env_name.replace("Vision", ""), "demo", f"returns_seed{args.trial_seed}.pkl"), "rb") as f:
            expected_returns = pickle.load(f)["cp_returns"].mean(0)

        expected_returns *= args.epsilon
        model.set_expected_return_init(expected_returns)

    # save initial model
    model.save(os.path.join(log_path, f"rl_model_seed{args.trial_seed}_0_steps.zip"))

    print(model.policy)

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq = args.save_freq,
        save_path = log_path,
        name_prefix = f"rl_model_seed{args.trial_seed}",
        save_replay_buffer=True
    )

    model.learn(args.total_timesteps, callback=checkpoint_callback)

    print("Time:", time.time() - start_time)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", help="Environment name.")
    parser.add_argument("--exp_name", default="")

    parser.add_argument("--epsilon", type=float, default=1.0, help="constant to multiply with expected return of initial pi.")

    parser.add_argument("--total_timesteps", default=100000, type=int, help="Total number of timesteps to train")
    parser.add_argument("--save_freq", default=5000, type=int, help="Frequency of saving model checkpoint")

    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")

    parser.add_argument("--mode", type=str, default="dist", 
        help="VaR PPO Mode. 'dist' for optimizing wrt to reward distribution, 'mean' for \
        optimizing wrt to mean reward, 'true' for optimizing wrt to true reward (same as using PPO).")
    parser.add_argument("--alpha", type=float, default=0.975, help="alpha of Value-at-Risk")
    
    parser.add_argument("--trial_seed", type=int, default=1, help="Seed to determine which trial [1-20].")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")

    args = parser.parse_args()

    candidate_selection(args)

