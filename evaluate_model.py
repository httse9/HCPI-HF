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

def create_env(args):
    env = gym.make(args.env_name)
    samples_filename = os.path.join(f"samples_stateFeatAvail_seed{args.seed}.npy")
    env = SafetyGoalFeatureWrapper(env, args.env_name, samples_filename=samples_filename, mode="dist")
    env = Monitor(env)
    return env

def evaluate_model(args, verbose=True):
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    env = create_env(args)
    n_rewards = env.get_n_rewards()

    if args.model_type == "ppo":
        model = VaRPPO.load(args.model_path, n_rewards)
    elif args.model_type == "our" or args.model_type == "baseline":
        path = os.path.join(args.model_path, f"rl_model_seed{args.seed}_{args.total_timesteps}_steps.zip")
        model = VaRPPO.load(path, n_rewards)
    elif args.model_type == "cpl":
        model = CPLModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, f"model_seed{args.seed}.pt"),))

    # returns used to evaluate the model
    eval_returns = evaluate_policy_reward_dist(model, env, n_rewards, n_eval_episodes=args.n_eval_episodes, deterministic=args.deterministic)
    eval_returns = eval_returns.mean(0)
    print(eval_returns.shape)       # (# reward samples + 1, ), + 1 because of ground truth

    data = dict(
        eval_returns=eval_returns,
    )

    if args.test:
        # returns used for safety test 
        test_returns = evaluate_policy_reward_dist(model, env, n_rewards, n_eval_episodes=args.n_test_episodes, deterministic=args.deterministic)
        data["test_returns"] = test_returns
        print(test_returns.shape)       # (# test episodes, # reward samples + 1)

    path = os.path.join("threshold", args.env_name, args.thres_name)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f"returns_seed{args.seed}.pkl"), "wb") as f:
        pickle.dump(data, f)

    print("Time:", time.time() - start_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("model_type", type=str, help="Either 'our' for our algorithm, 'ppo' for demo policy, or 'baseline' for baselines")
    parser.add_argument("thres_name", help="name of threshold")
    parser.add_argument("model_path", help="Path to the initial policy.")
    parser.add_argument("--n_eval_episodes", type=int, default=200, help="Number of episodes to evaluate policies for true return.")
    parser.add_argument("--n_test_episodes", type=int, default=2000, help="Number of episodes to use for safety test.")
    parser.add_argument("--total_timesteps", default=5000000, type=int, help="Total number of timesteps to train")
    parser.add_argument("--seed", type=int, default=0, help="Random seed of candidate policy to test")
    parser.add_argument("--test", action="store_true", help="Whether to generate test episodes.")
    parser.add_argument("--deterministic", action="store_true", default=False)
    
    args = parser.parse_args()
    evaluate_model(args)