import torch
import torch.nn as nn
from torch.distributions import Normal
import safety_gymnasium as gym
import torch.nn.functional as F
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.utils import set_random_seed

"""
For the safety gymnaisum domains we consider, the true reward function
can be expressed linearly in the state features. Here we use the state features
as input, and learn a linear model mapping from state features to rewards
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_feat_label(args, seed):
    with open(os.path.join("./preferences", args.env_name, f"preferences_seed{seed}.pkl"), 'rb') as f:
        preferences = pickle.load(f)


    feat = preferences["feat"]          # (n_demo, 2, episode_length, state_feature_dim)
    label = preferences["label"]      # (n_demo, )

    feat = torch.tensor(feat, device=device, dtype=torch.float).sum(2)    # (n_demo, 2, state_feature_dim)
    label = torch.tensor(label, device=device).long()         # (n_demo, )

    return feat, label

def train_trex_model(args, verbose=False, test=True):

    feat, label = load_feat_label(args, args.seed)

    if test:
        test_feat, test_label = load_feat_label(args, args.seed + 1)

    cross_entropy_loss = nn.CrossEntropyLoss()

    reward_dim = int(feat.size(-1))
    trex_reward = torch.rand(reward_dim, device=device, requires_grad=True) 

    optimizer = torch.optim.Adam([trex_reward], lr=args.lr)

    losses = []
    test_losses = []

    for e in range(args.n_epochs):


        returns = torch.matmul(feat, trex_reward)       # (n_demo, 2)
        loss = cross_entropy_loss(returns, label)

        losses.append(loss.item())
        if verbose:
            print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if test:
            with torch.no_grad():
                test_returns = torch.matmul(test_feat, trex_reward)
                test_loss = cross_entropy_loss(test_returns, test_label)
                test_losses.append(test_loss.item())

    learned_reward = trex_reward.detach().clone().cpu().numpy()
    # normalize
    learned_reward /= np.sqrt(np.square(learned_reward).sum())

    return learned_reward, losses, test_losses

    

    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning Rate")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    ### hyperparameter tuning
    # for lr in [1e-2, 3e-2, 1e-1, 3e-1, 1]:
    #     loss_list = []
    #     for seed in range(1, 5 + 1):
    #         set_random_seed(seed)
    #         args.lr = lr
    #         args.seed = seed
    #         reward, losses, test_losses = train_trex_model(args)
    #         loss_list.append(test_losses)

    #     loss_list = np.array(loss_list).mean(0)
    #     plt.plot(loss_list, label=str(lr))

    # plt.legend()
    # plt.savefig("trex_circle_lr.png")
    ### Best learing rate: 1e-1

    ### check overfitting
    # loss_list = []
    # test_loss_list = []
    # for seed in range(1, 5 + 1):
    #     args.seed = seed
    #     reward, losses, test_losses = train_trex_model(args)
    #     loss_list.append(losses)
    #     test_loss_list.append(test_losses)
    # plt.plot(np.array(loss_list).mean(0), label="train")
    # plt.plot(np.array(test_loss_list).mean(0), label="test")
    # plt.legend()
    # plt.savefig("trex_cirlce.png")
    ### slight overfitting, shoudn't be a big issue.

    ### Plot the reward functions learned
    # for seed in range(1, 5 + 1):
    #     args.seed = seed
    #     reward, _, _ = train_trex_model(args)
    #     plt.plot(reward)
    # plt.savefig("trex_circle_rewards.png")

    ### Learn the reward functions
    reward, _, _ = train_trex_model(args, test=False)

    save_path = os.path.join("reward_samples", args.env_name)
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, f"trex_reward_seed{args.seed}.pkl"), "wb") as f:
        pickle.dump(reward, f)
