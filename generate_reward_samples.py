import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import safety_gymnasium as gym
from env_wrapper import SafetyGoalFeatureWrapper

def bayesian_rex(args, verbose=True):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load reward model training data
    with open(os.path.join("./preferences", args.env_name, f"preferences_seed{args.seed}.pkl"), 'rb') as f:
        preferences = pickle.load(f)

    feat = preferences["feat"]
    label = preferences["label"]

    feat = torch.tensor(feat, device=device).float()
    label = torch.tensor(label, device=device).long()
    #  obs: (n_demo, 2, n_timesteps, state_feature_dim)

    feat = feat.sum(dim=2)  # (n_demo, 2, state_feature_dim)

    """
    Distributions
    reward: torch tensor (encode_dim,)
    """
    def prior(reward, feat):
        # uniform prior
        return 1

    def log_likelihood(reward, feat, label, beta=args.beta):
        # beta: confidence in demo, should match data generation
        returns = torch.matmul(feat, reward) * beta    # (n_demo, 2)
        return - nn.CrossEntropyLoss(reduction="sum")(returns, label)
        
    def proposal(reward, sigma=args.proposal_width):
        proposal = reward + torch.randn(reward.size()).to(device) * sigma
        # normalize l2 norm
        proposal /= torch.sqrt(torch.square(proposal).sum())
        return proposal

    """
    MCMC
    """
    if verbose:
        print("**********************")
        print("Running MCMC......")

    # init reward
    r = proposal(torch.zeros(feat.size(2)).to(device))
    ll = log_likelihood(r, feat, label)

    burnin = int(args.burnin_ratio * args.n_mcmc_steps)

    samples = []
    lls = []
    acceptance_count = 0
    for t in range(args.n_mcmc_steps):

        r_proposal = proposal(r)
        ll_proposal = log_likelihood(r_proposal, feat, label)

        # acceptance condition
        accept = False
        if prior(r_proposal, feat) > 0:
            if ll_proposal > ll:
                # proposal has higher prob than orig, accept
                accept = True
            elif np.random.uniform() < torch.exp(ll_proposal - ll):
                # proposal has lower prob, accept with prob ratio
                accept = True
        
        if accept:
            r = r_proposal
            ll = ll_proposal
            acceptance_count += 1
        
        if t >= burnin and t % args.skip_interval == 0:
            samples.append(r.view(1, -1))
            lls.append(ll.item())

    if verbose:
        print("Acceptance ratio:", acceptance_count / args.n_mcmc_steps)

    # split into two sets, half for candidate proposal and half for safety test
    samples_cp = [s for i, s in enumerate(samples) if i % 2 == 0]
    samples_st = [s for i, s in enumerate(samples) if i % 2 == 1]

    samples_cp = torch.cat(samples_cp, dim=0).cpu().numpy()
    samples_st = torch.cat(samples_st, dim=0).cpu().numpy()

    lls_cp = [ll for i, ll in enumerate(lls) if i % 2 ==0]

    # MAP reward
    map_r = samples_cp[np.argmax(lls_cp)]
    # print(map_r)

    # mean reward
    mean_r = samples_cp.mean(0)
    # print(mean_r)

    if verbose:
        print(len(samples), "samples collected.")
        print("Finished.")
        print("**********************")

    # save samples
    save_path = os.path.join("./reward_samples", args.env_name)
    os.makedirs(save_path, exist_ok=True)

    filename = f"samples_seed{args.seed}.pkl"

    data = dict(
        samples=samples_cp,    # refer to samples for candidate proposal
        mean_r=mean_r,
        map_r=map_r,
        test_samples=samples_st
    )

    with open(os.path.join(save_path, filename), "wb") as f:
        pickle.dump(data, f)

    ### plot reward samples
    # make plots directory
    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
        
    # plot
    samples = torch.cat(samples, dim=0).cpu().numpy()
    for s in samples[:-1]:
        plt.plot(s, alpha=0.05, color='blue')
    plt.plot(samples[-1], alpha=0.05, color="blue", label="samples")

    env = gym.make(args.env_name)
    true_reward = SafetyGoalFeatureWrapper(env, args.env_name).get_true_reward()
    # normalize to have L2 norm
    true_reward /= np.sqrt(np.square(true_reward).sum())
    plt.plot(true_reward, color="red", label="true")

    plt.plot(mean_r, color="green", label="mean")
    plt.plot(map_r, color="magenta", label="map")
    plt.axhline(0, color='black')
    plt.legend()

    filename = f"samples_seed{args.seed}.png"
    plt.savefig(os.path.join(plot_dir, filename))

    return mean_r, map_r, samples


if __name__ == "__main__":
    import argparse
    from stable_baselines3.common.utils import set_random_seed
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--n_mcmc_steps", type=int, default=20000, help="Number of MCMC steps")
    parser.add_argument("--burnin_ratio", type=float, default=0.2, help="Number of MCMC steps to use for burnin.")
    parser.add_argument("--skip_interval", type=int, default=20, help="How many proposals to skip between taking\
        a sample to reduce auto-correlation.")
    parser.add_argument("--proposal_width", type=float, default=1, help="Proposal width. Should adjust so that\
        acceptance probability is in reasonable range.")
    parser.add_argument("--beta", type=float, default=5, help="Temperature paramter in Bradley-Terry model.")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")

    args = parser.parse_args()
    set_random_seed(args.seed)
    bayesian_rex(args)