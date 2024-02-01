import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from sklearn.utils import resample

import matplotlib
matplotlib.rcParams.update({'font.size': 13})

def HCLB(data, delta):
    # calculate high confidence lower bound for each row of data
    # using t test
    m = data.shape[1]
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1, ddof=1)

    return mean - std / np.sqrt(m) * stats.t.ppf(1 - delta, m - 1)

def compute_prob(args):
    path = os.path.join("threshold", args.env_name)

    # load trex
    trex_returns = []
    for s in range(1, 21):
        with open(os.path.join(path, "trex", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        trex_returns.append(data["eval_returns"][-1])
    trex_returns = np.array(trex_returns)

    # load brex mean
    brex_mean_returns = []
    for s in range(1, 21):
        with open(os.path.join(path, "mean", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        brex_mean_returns.append(data["eval_returns"][-1])
    brex_mean_returns = np.array(brex_mean_returns)

    # load brex map
    brex_map_returns = []
    for s in range(1, 21):
        with open(os.path.join(path, "map", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        brex_map_returns.append(data["eval_returns"][-1])
    brex_map_returns = np.array(brex_map_returns)

    # load pgbroil
    pgbroil_returns = []
    pgbroil_dist_returns = []
    for s in range(1, 21):
        with open(os.path.join(path, "pgbroil", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        pgbroil_returns.append(data["eval_returns"][-1])
        pgbroil_dist_returns.append(data['test_returns'][:, :-1])
    pgbroil_returns = np.array(pgbroil_returns)
    pgbroil_dist_returns = np.array(pgbroil_dist_returns)

    # load cpl
    cpl_returns = []
    for s in range(1, 21):
        with open(os.path.join(path, "cpl", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        cpl_returns.append(data['eval_returns'][-1])
    cpl_returns = np.array(cpl_returns)

    # load optimal
    optimal_returns = []
    optimal_dist_returns = []
    for s in range(1, 21):
        with open(os.path.join(path, "optimal", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        optimal_returns.append(data['eval_returns'])
        optimal_dist_returns.append(data['test_returns'][:, :-1])

    optimal_returns = np.array(optimal_returns)
    optimal_dist_returns = np.array(optimal_dist_returns)

    colors = ["r", "g", "b", "magenta", "brown"]
    
    for e, color in zip(epsilon, colors):
        print("Epsilon:", e)

        # optimal * epsilon
        opt_returns = optimal_returns * e
        opt_return = opt_returns[:, -1].mean()  
        # print("Return of initial policy:", opt_return)
        opt_dist_returns = optimal_dist_returns * e

        # load POSTPI
        if not args.vision:
            our_returns = []
            our_dist_returns = []
            for s in range(1, 21):
                with open(os.path.join(path, f"thres_init_opt{e}", f"returns_seed{s}.pkl"), "rb") as f:
                    data = pickle.load(f)
                our_returns.append(data['eval_returns'][-1])
                our_dist_returns.append(data['test_returns'][:, :-1])
            our_returns = np.array(our_returns)
            our_dist_returns = np.array(our_dist_returns)
        else:
            # load ours vision
            our_vision_returns = []
            our_vision_dist_returns = []
            for s in range(1, 21):
                with open(os.path.join("threshold", args.env_name.replace("-", "Vision-"), f"thres_init_opt{e}_5M", f"returns_seed{s}.pkl"), "rb") as f:
                    data = pickle.load(f)
                our_vision_returns.append(data['eval_returns'][-1])
                our_vision_dist_returns.append(data['test_returns'][:, :-1])
            our_returns = np.array(our_vision_returns)
            our_dist_returns = np.array(our_vision_dist_returns)

        # num episodes to use in safey test
        n_episodes = min(our_dist_returns.shape[1], opt_dist_returns.shape[1])
        ns = np.linspace(2, n_episodes, num=20)
        ns = [int(n) for n in ns]

        # our safety test
        prob_accept_our = []
        prob_improv_our = []

        # b-rex style safety test
        prob_accept_hcpe = []
        prob_improv_hcpe = []

        # exp. return of accepted policies
        return_accepted = []

        for s in range(1, 21):
            # print("Seed:", s)

            # compute estimates of J(pi_C, r) - J(pi_init, r)
            return_difference = (our_dist_returns[s - 1] - opt_returns[s - 1, :-1]).T       # (n_rewards, n_episodes)


            accept_our = []
            improv_our = []

            accept_hcpe = []
            improv_hcpe = []

            for n in ns:
                ###### our safety test
                lower_bounds = HCLB(return_difference[:, :n], args.delta / 2)
                L = np.quantile(lower_bounds, args.delta / 2, method="lower")

                accept_our.append(int(L >= 0))
                if accept_our[-1] == 1:
                    # pi_C accepted, compare returns
                    improv_our.append(int(our_returns[s-1] >= opt_returns[s - 1, -1]))
                    return_accepted.append(our_returns[s - 1])
                else:
                    # pi_C rejected, safe
                    improv_our.append(1)

                ###### see Bayesian REX paper Section 5.3
                hcpe_estimates = return_difference[:, :n].mean(1)
                L = np.quantile(hcpe_estimates, args.delta, method="lower")

                accept_hcpe.append(int(L >= 0))

                if accept_hcpe[-1] == 1:
                    improv_hcpe.append(int(our_returns[s - 1] >= opt_returns[s - 1, -1]))
                else:
                    improv_hcpe.append(1)

            prob_accept_our.append(accept_our)
            prob_improv_our.append(improv_our)

            assert len(accept_hcpe) == len(ns)

            prob_accept_hcpe.append(accept_hcpe)
            prob_improv_hcpe.append(improv_hcpe)
        
        # print("Average returns of accepted policies:", np.mean(return_accepted))

        prob_accept_our = np.array(prob_accept_our).mean(0)
        prob_improv_our = np.array(prob_improv_our).mean(0)
        # prob_improv = np.array(prob_improv).sum(0) / sum_accept

        prob_accept_hcpe = np.array(prob_accept_hcpe).mean(0)
        prob_improv_hcpe = np.array(prob_improv_hcpe).mean(0)


        #### print 
        # our improvement
        our_improv = prob_improv_our.min()      # min over the num episodes used in safety test
        print("  Our Improv:", our_improv, np.mean(return_accepted))

        # hcpe improvement
        hcpe_improv = prob_improv_hcpe.min()
        print("  HCPE Improv:", hcpe_improv)

        # trex improvement
        trex_improv = (trex_returns > opt_return).astype(int).mean()
        print("  TREX Improv:", trex_improv, trex_returns.mean())

        # brex mean improvement
        brex_mean_improv = (brex_mean_returns > opt_return).astype(int).mean()
        print("  BREX MEAN Improv:", brex_mean_improv, brex_mean_returns.mean())

        # brex map improvement
        brex_map_improv = (brex_map_returns > opt_return).astype(int).mean()
        print("  BREX MAP Improv:", brex_map_improv, brex_map_returns.mean())

        # pgbroil improvement
        pgbroil_improv = (pgbroil_returns > opt_return).astype(int).mean()
        print("  PGBROIL Improv:", pgbroil_improv, pgbroil_returns.mean())

        # cpl improvement
        cpl_improv = (cpl_returns > opt_return).astype(int).mean()
        print("  CPL Improv:", cpl_improv, cpl_returns.mean())



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--delta", type=float, default=0.05, help="Desired confidence level is (1 - delta).")
    parser.add_argument("--vision", action='store_true')
    args = parser.parse_args()

    epsilon = [0, 0.25, 0.5, 0.75, 1]

    compute_prob(args)