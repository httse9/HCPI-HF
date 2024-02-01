import pickle
import os
import torch
from stable_baselines3.ppo.var_ppo import VaRPPO


def generate_labels(args, verbose=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load data
    pref_path = os.path.join("reward_models", args.env_name, "reward_model_data.pkl")
    with open(pref_path, "rb") as f:
        data = pickle.load(f)

    obs = data["obs"]           # (n_pairs, 2, episode_length, obs_dim)
    rew = data["reward"]        # (n_pairs, 2, episode_length)

    obs = torch.tensor(obs, device=device).float()
    rew = torch.tensor(rew, device=device).float()

    # load model
    assert "true" in args.model_path        # loading model trained under ground truth reward
    model = VaRPPO.load(args.model_path, 1)

    # approx advantage      (r + gamma * V(obs') - V(obs))
    _, V, _ = model.policy(obs.view(-1, obs.size(-1)))
    V = V.view(*obs.size()[:-1])        # (n_pairs, 2, episode_length)

    # args.gamma is the gamma used for computing advantage, not gamma used in CPL paper
    # args.gamma follows the one used when training demo policy, which is 0.99
    A = rew[:, :, 1:] + args.gamma *  V[:, :, 1:] - V[:, :, :-1]    # (n_pairs, 2, episode_length - 1)

    # here we assume gamma of CPL paper = 1
    A = A.sum(-1)   # (n_pairs, 2)

    A *= args.beta

    # preference model
    prob_0_better = 1 / (1 + torch.exp(A[:, 1] - A[:, 0]))
    # print(prob_0_better.size())       # (n_pairs,)

    label = (torch.rand(prob_0_better.size(), device=device) >= prob_0_better).int()
    label = label.detach().cpu().numpy()
    # print(label.shape)      # (n_pairs, )

    data['label'] = label

    with open(os.path.join("reward_models", args.env_name, "CPL_data.pkl"), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", help="Environment name.")
    parser.add_argument("model_path")
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--beta", default=1, type=float)

    args = parser.parse_args()
    generate_labels(args)