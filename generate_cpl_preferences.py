import pickle
import os
import torch
from stable_baselines3.ppo.var_ppo import VaRPPO
from stable_baselines3 import PPO

"""
Problem with VaRPPO load. Should change to PPO load?

In VarPPO load, when we train algorithms with only one reward, such as B-REX and T-REX,
they have saved parameters such that the value network has output dimension of 1.

However, during evaluation, when we want to evaluate them for a bunch of reward samples,
we need to set the n_rewards to something like 400, e.g., to set the correct replay buffer size.
Then, there's a mismatch between the dimensions of the newly intiailized value network and the 
original value network saved in a checkpoint. That's why I overrode the n_rewards in the VaRPPO load code,
and also disabled loading parameters of the value network.

Here, the preferences generated are probably wrong, since we are using VaRPPO.load
The value network becomes some randomly initialized thing, that's super wrong.
"""

gae_lambda = 0.95
gamma = 0.99

def generate_labels(args, verbose=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load data
    pref_path = os.path.join("preferences", args.env_name, f"preferences_seed{args.seed}.pkl")
    with open(pref_path, "rb") as f:
        preferences = pickle.load(f)

    obs = preferences["obs"]           # (n_pairs, 2, episode_length, obs_dim)
    rew = preferences["reward"]        # (n_pairs, 2, episode_length)

    obs = torch.tensor(obs, device=device).float()
    rew = torch.tensor(rew, device=device).float()

    # load model
    assert "true" in args.model_path        # loading model trained under ground truth reward
    model = PPO.load(args.model_path)       # !!! Do NOT use VaRPPO.load here
    # print(model.policy.value_net.weight.data)

    # approx advantage      (r + gamma * V(obs') - V(obs))
    _, V, _ = model.policy(obs.view(-1, obs.size(-1)))
    V = V.view(*obs.size()[:-1])        # (n_pairs, 2, episode_length)

    ### Advantage estimation 2, following PPO's method
    advantage = torch.zeros(rew.size(), device=device)
    episode_length = rew.size(-1)
    last_gae_lam = 0
    for step in reversed(range(episode_length)):
        if step == episode_length - 1:
            next_values = 0
        else:
            next_values = V[:, :, step + 1]
        
        delta = rew[:, :, step] + gamma * next_values - V[:, :, step]
        last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
        advantage[:, :, step] = last_gae_lam
    A = advantage

    ### Advantage estimation 1, r + gamma * V(s') - V(s)
    # gamma follows the one used when training demo policy, which is 0.99
    # V = torch.cat([V, torch.zeros(*V.size()[:-1], 1, device=device)], dim=-1)
    # A = rew + gamma *  V[:, :, 1:] - V[:, :, :-1]    # (n_pairs, 2, episode_length - 1)

    # here we assume gamma of CPL paper = 1
    A = A.sum(-1)   # (n_pairs, 2)

    # preference model
    prob_0_better = 1 / (1 + torch.exp(A[:, 1] - A[:, 0]))
    # print(prob_0_better.size())       # (n_pairs,)

    label = (torch.rand(prob_0_better.size(), device=device) >= prob_0_better).int()
    label = label.detach().cpu().numpy()
    # print(label.shape)      # (n_pairs, )

    preferences['cpl_label'] = label

    with open(os.path.join("preferences", args.env_name, f"cpl_preferences_seed{args.seed}.pkl"), "wb") as f:
        pickle.dump(preferences, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", help="Environment name.")
    parser.add_argument("model_path")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    generate_labels(args)


# python generate_cpl_preferences.py SafetyPointGoal5-v0 candidate_selection/SafetyPointGoal5-v0/true-0.0001-1/rl_model_seed1_5000000_steps.zip --seed 1
# python train_cpl.py SafetyPointGoal5-v0 --seed 1
# python evaluate_model.py SafetyPointGoal5-v0 cpl candidate_selection/SafetyPointGoal5-v0/cpl/ --seed 1