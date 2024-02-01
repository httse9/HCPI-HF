import torch
import torch.nn as nn
from torch.distributions import Normal
from generate_data import generate_reward_model_training_data
import safety_gymnasium as gym
import torch.nn.functional as F
import pickle
import os
import matplotlib.pyplot as plt

"""
For the safety gymnaisum domains we consider, the true reward function
can be expressed linearly in the state features. Here we use the state features
as input, and learn a linear model mapping from state features to rewards
"""

def train_trex_model(args, verbose=True):

    with open(os.path.join("reward_models", args.env_name, "reward_model_data_stateFeatAvail.pkl"), "rb") as f:
        training_data = pickle.load(f)

    obs = training_data["obs"]          # (n_demo, 2, episode_length, state_feature_dim)
    label = training_data["label"]      # (n_demo, )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    obs = torch.tensor(obs, device=device, dtype=torch.float).sum(2)    # (n_demo, 2, state_feature_dim)
    label = torch.tensor(label, device=device).long()         # (n_demo, )

    cross_entropy_loss = nn.CrossEntropyLoss()

    reward_dim = int(obs.size(-1))
    trex_reward = torch.rand(reward_dim, device=device, requires_grad=True) 

    optimizer = torch.optim.Adam([trex_reward], lr=args.lr)

    losses = []

    for e in range(args.n_epochs):


        returns = torch.matmul(obs, trex_reward)       # (n_demo, 2)
        loss = cross_entropy_loss(returns, label)

        losses.append(loss.item())
        if verbose:
            print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    learned_reward = trex_reward.detach().clone().cpu().numpy()

    save_path = os.path.join("reward_samples", args.env_name)
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, f"trex_reward_seed{args.seed}.pkl"), "wb") as f:
        pickle.dump(learned_reward, f)

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("TREX Loss")
    plt.savefig(os.path.join(save_path, f"trex_reward_seed{args.seed}.png"))

    

if __name__ == "__main__":
    import argparse
    from stable_baselines3.common.utils import set_random_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning Rate")
    parser.add_argument("--n_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    set_random_seed(args.seed)
    train_trex_model(args)