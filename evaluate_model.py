from env_wrapper import SafetyGoalFeatureWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy_reward_dist
from stable_baselines3.ppo.var_ppo import VaRPPO
import os
import safety_gymnasium as gym
import pickle
import time
from train_cpl import CPLModel
import torch

os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_env(args):
    env = gym.make(args.env_name)

    samples_filename = os.path.join(f"samples_seed{args.seed}.pkl")

    # for algorithms, evalaute on the set of reward samples reserved for safety test
    samples_type = ["st"]

    # for demo policy, evaluate on both sets of reward samples
    if args.mode == "demo":
        samples_type = ["cp", "st"]

    env = SafetyGoalFeatureWrapper(env, args.env_name, samples_filename=samples_filename, mode="dist",
                                   samples_type=samples_type)
    env = Monitor(env)
    return env

def evaluate_demo_policy(args, verbose=True):
    env = gym.make(args.env_name)
    samples_filename = os.path.join(f"samples_seed{args.seed}.pkl")

    env = SafetyGoalFeatureWrapper(env, args.env_name, samples_filename=samples_filename, mode="dist",
                                   samples_type=["cp", "st"])
    env = Monitor(env)
    n_rewards = env.get_n_rewards()
    
    start_time = time.time()

    model = VaRPPO.load(args.model_path, n_rewards)

    eval_returns = evaluate_policy_reward_dist(model, env, n_rewards, n_eval_episodes=200, deterministic=False)

    cp_returns = eval_returns[:, :400]
    st_returns = eval_returns[:, 400: 800]
    true_return = eval_returns[:, -1]

    results = dict(
        cp_returns=cp_returns,
        st_returns=st_returns,
        true_return=true_return
    )

    path = os.path.join("evaluation", args.env_name, "demo")
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f"returns_seed{args.seed}.pkl"), "wb") as f:
        pickle.dump(results, f)

    print("Time:", time.time() - start_time)


def evaluate_model(args, verbose=True):
    
    start_time = time.time()

    # create env
    env = create_env(args)
    n_rewards = env.get_n_rewards()

    # load model
    
    if args.mode in ["dist", "pgbroil", "mean", "map", "trex"]:
        # path to checkpoint
        path = os.path.join(args.model_path, f"rl_model_seed{args.seed}_{args.total_timesteps}_steps.zip")
        model = VaRPPO.load(path, n_rewards)

    elif args.mode == "cpl":
        model = CPLModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, f"model_seed{args.seed}.pt"),))

    else:
        raise ValueError(f"Mode {args.mode} not recognized.")
    
    if args.mode in ['dist', 'pgbroil', "cpl"]:
        n_eval_episodes = 1000
    else:
        n_eval_episodes = 200

    # returns used to evaluate the model
    eval_returns = evaluate_policy_reward_dist(model, env, n_rewards, n_eval_episodes=n_eval_episodes, deterministic=False)

    cp_returns = eval_returns[:, :400]
    true_return = eval_returns[:, -1]

    results = dict(
        cp_returns=cp_returns,
        true_return=true_return
    )

    path = os.path.join("evaluation", args.env_name, args.mode)
    if args.mode == "dist":
        epsilon_path = args.model_path.split("/")[-1]
        if len(epsilon_path) == 0:
            epsilon_path = args.model_path.split("/")[-2]
        path = os.path.join(path, epsilon_path)

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f"returns_seed{args.seed}.pkl"), "wb") as f:
        pickle.dump(results, f)

    print("Time:", time.time() - start_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("mode", type=str, help="Either 'our' for our algorithm, 'ppo' for demo policy, or 'baseline' for baselines")
    parser.add_argument("model_path", help="Path to the initial policy.")


    parser.add_argument("--seed", type=int, default=0, help="Random seed of candidate policy to test")
    
    args = parser.parse_args()

    # if args.env_name == "SafetyPointGoal5-v0":
    if "Goal" in args.env_name:
        args.total_timesteps = 5000000
    # elif args.env_name == "SafetyPointCircle1-v0":
    elif "Circle" in args.env_name:
        args.total_timesteps = 3000000
    else:
        raise ValueError(f"Env name {args.env_name} not recognized.")

    if args.mode == "demo":
        evaluate_demo_policy(args)
    else:
        evaluate_model(args)
