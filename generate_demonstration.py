import safety_gymnasium as gym
import os
import numpy as np
import pickle
from env_wrapper import SafetyGoalFeatureWrapper
from stable_baselines3.ppo.var_ppo import VaRPPO


def generate_trajectory(env, model, verbose=True):
    """
    Generate one trajectory using model
    """
    obs, info = env.reset()

    ret = 0
    feat_list = []  # state features
    obs_list = []   # state
    act_list = []   # action
    rew_list = []   # reward

    done = False
    while not done:

        a, _ = model.predict(obs, deterministic=False)

        feat_list.append(info['state_features'])
        obs_list.append(obs)
        act_list.append(a)
        
        obs, r, terminated, truncated, info = env.step(a)

        ret += r
        rew_list.append(r)

        if terminated or truncated:
            break

    features = np.array(feat_list)
    observations = np.array(obs_list)
    actions = np.array(act_list)
    rewards = np.array(rew_list)
    
    print(features.shape)
    print(observations.shape)
    print(actions.shape)
    print(ret)

    if verbose:
        print("  Trajectory generated with return:", ret)

    return (features, observations, actions, rewards), ret

def generate_demo(env, model, args, verbose=True):
    demo_count = 0

    episodes = []
    returns = []

    while demo_count < args.n_demo:

        episode, ret = generate_trajectory(env, model, verbose=verbose)

        # discard episodes with return below threshold
        if ret < args.return_thres:
            continue

        episodes.append(episode)
        returns.append(ret)
        demo_count += 1

    return episodes, returns
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--n_demo", type=int, default=10, help="Number of demonstration")
    parser.add_argument("--return_thres", default=None, type=float, help="Only use episodes with return above this threshold.")

    args = parser.parse_args()

    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(42)


    env = gym.make(args.env_name)
    env = SafetyGoalFeatureWrapper(env, args.env_name, mode="true")
    n_rewards = env.get_n_rewards()

    if args.env_name == "SafetyPointGoal5-v0":
        model_path = "candidate_selection/SafetyPointGoal5-v0/true-0.0001-1/rl_model_seed1_5000000_steps.zip"
    elif args.env_name == 'SafetyPointCircle1-v0':
        model_path = "candidate_selection/SafetyPointCircle1-v0/true-0.0001-0/rl_model_seed0_3000000_steps.zip"

    model = VaRPPO.load(model_path, n_rewards)

    episodes, returns = generate_demo(env, model, args)

    demo = dict(episodes=episodes, returns=returns)

    write_path = os.path.join("./demo", args.env_name)
    os.makedirs(write_path, exist_ok=True)


    with open(os.path.join(write_path, "demo.pkl"), "wb") as f:
        pickle.dump(demo, f)
    