import safety_gymnasium as gym
from stable_baselines3 import PPO
import os
import numpy as np
import pickle
from stable_baselines3.common.utils import set_random_seed

def generate_random_index_pair(n_demo):
    """
    Generate a pair of random indices i, j,
    where i not equal to j.
    """
    i = np.random.randint(0, n_demo)
    j = np.random.randint(0, n_demo)

    while (i == j):
        j = np.random.randint(0, n_demo)

    return i, j

def generate_preferences(args):
    demo_path = os.path.join("./demo", args.env_name, "demo.pkl")
    with open(demo_path, "rb") as f:
        demo = pickle.load(f)

    episodes = demo['episodes']
    returns = demo['returns']
    n_episodes = len(returns)

    wrong_label_count = 0

    training_feat = []
    training_obs = []
    training_act = []
    training_reward = []
    training_label = []

    for _ in range(args.n_demo):
        i, j = generate_random_index_pair(n_episodes)

        # compute preference label using bradley terry model
        prob_i_better_than_j = 1 / (1 + np.exp(args.beta * (returns[j] - returns[i])))
        if np.random.uniform() < prob_i_better_than_j:
            label = 0

            if returns[j] > returns[i]:
                wrong_label_count += 1
        else:
            label = 1

            if returns[i] > returns[j]:
                wrong_label_count += 1

        training_feat.append([episodes[i][0], episodes[j][0]])
        training_obs.append([episodes[i][1], episodes[j][1]])
        training_act.append([episodes[i][2], episodes[j][2]])
        training_reward.append([episodes[i][3], episodes[j][3]])
        training_label.append(label)

    print("wrong label count", wrong_label_count)

    training_feat = np.array(training_feat)
    training_obs = np.array(training_obs)
    training_act = np.array(training_act)
    training_reward = np.array(training_reward)
    training_label = np.array(training_label)

    # print(training_feat.shape, training_obs.shape, training_reward.shape, training_label.shape)
    # (n_demo, 2, episode_length, feat/obs/reward/label dim)

    preference_data = {
        "feat": training_feat,
        "obs": training_obs,
        "act": training_act,
        "reward": training_reward,
        "label": training_label
    }

    save_path = os.path.join("./preferences", args.env_name)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, f"preferences_seed{args.seed}.pkl"), "wb") as f:
        pickle.dump(preference_data, f)

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--n_demo", type=int, default=50, help="Number of demonstration")
    parser.add_argument("--beta", type=float, default=5, help="Inverse temperature paramter in Bradley-Terry model.")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    set_random_seed(args.seed)
    generate_preferences(args)
