import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

import matplotlib
matplotlib.rcParams.update({'font.size': 13})

def HCLB(data, delta):
    # calculate high confidence lower bound for each row of data
    # using t test
    m = data.shape[1]
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1, ddof=1)

    return mean - std / np.sqrt(m) * stats.t.ppf(1 - delta, m - 1)

def load_true_return(env_name, mode):
    path = os.path.join("evaluation", env_name, mode)

    returns = []
    for s in range(1, 21):
        with open(os.path.join(path, f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        returns.append(data['true_return'].mean())

    return np.array(returns)

def postpi_safety_test(env_name, epsilon, n_eps = None):

    accept_list = []
    worse_list = []
    true_return_list = []
    actual_better = []

    for s in range(1, 21):
        with open(os.path.join("evaluation", env_name.replace("Vision", ""), "demo", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)

        demo_returns = data['st_returns'].mean(0).reshape(1, -1) * epsilon
        demo_true_return = data['true_return'].mean() * epsilon

        # print(demo_returns.shape)   # (1, 400)
        # print(demo_true_return)

        # load postpi returns
        try:
            path = os.path.join("evaluation", env_name, "dist", f"epsilon{epsilon}")
            with open(os.path.join(path, f"returns_seed{s}.pkl"), "rb") as f:
                data = pickle.load(f)
        except:
            print("Cannot load data", env_name, epsilon, s)
            continue

        # expected return of postpi policy under ground-truth reward function
        # for trial $s of epsilon $epsilon
        true_return = data['true_return'].mean()
        true_return_list.append(true_return)

        # in evaluate_model.py, typo: cp_returns instead of st_returns
        # but reward samples and values are correct, just the key name wrong
        returns = data['cp_returns']
        # print(returns.shape)     # (n_eval_episodes, n_rewards = 400)

        return_difference = (returns - demo_returns).T

        # test different number of episodes used for safey test
        if n_eps is None:
            n_eps = np.linspace(2, min(returns.shape[0], 1000), 20)
            n_eps = [int(n) for n in n_eps]

        accept = []     # 1 if policy is accepted
        worse = []      # 1 if accepted policy is worse than initial policy
        for n in n_eps:

            lower_bounds = HCLB(return_difference[:, :n], args.delta / 2)      # (n_rewards, )
            L = np.quantile(lower_bounds, args.delta / 2, method="lower")

            if L >= 0:
                # accept
                accept.append(1)
                worse.append(int(true_return < demo_true_return))

                if true_return > demo_true_return:
                    actual_better.append(1)
                else:
                    actual_better.append(0)
            else:
                # reject
                accept.append(0)
                worse.append(0)     # rejected, so safe

        accept_list.append(accept)
        worse_list.append(worse)

    # average over 20 trials
    accept_list = np.array(accept_list)         # used to plot acceptance probability
    worse_list = np.array(worse_list).mean(0)       # take maximum, used in tables

    # print((true_return_list))
    print(">", np.mean(actual_better))

    return worse_list.max(), accept_list, worse_list, n_eps


def baseline_safety_test(env_name, epsilon, n_eps = None, alg="pgbroil"):

    accept_list = []
    worse_list = []
    true_return_list = []

    for s in range(1, 21):
        with open(os.path.join("evaluation", env_name.replace("Vision", ""), "demo", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)

        demo_returns = data['st_returns'].mean(0).reshape(1, -1) * epsilon
        demo_true_return = data['true_return'].mean() * epsilon

        # print(demo_returns.shape)   # (1, 400)
        # print(demo_true_return)

        # load postpi returns
        try:
            path = os.path.join("evaluation", env_name, alg)
            with open(os.path.join(path, f"returns_seed{s}.pkl"), "rb") as f:
                data = pickle.load(f)
        except:
            print("Cannot load data", env_name, epsilon, s)
            continue

        # expected return of postpi policy under ground-truth reward function
        # for trial $s of epsilon $epsilon
        true_return = data['true_return'].mean()
        true_return_list.append(true_return)

        # in evaluate_model.py, typo: cp_returns instead of st_returns
        # but reward samples and values are correct, just the key name wrong
        returns = data['cp_returns']
        # print(returns.shape)     # (n_eval_episodes, n_rewards = 400)

        return_difference = (returns - demo_returns).T

        # test different number of episodes used for safey test
        if n_eps is None:
            n_eps = np.linspace(2, min(returns.shape[0], 1000), 20)
            n_eps = [int(n) for n in n_eps]

        accept = []     # 1 if policy is accepted
        worse = []      # 1 if accepted policy is worse than initial policy
        for n in n_eps:

            lower_bounds = HCLB(return_difference[:, :n], args.delta / 2)      # (n_rewards, )
            L = np.quantile(lower_bounds, args.delta / 2, method="lower")

            if L >= 0:
                # accept
                accept.append(1)
                worse.append(int(true_return < demo_true_return))
            else:
                # reject
                accept.append(0)
                worse.append(0)     # rejected, so safe

        accept_list.append(accept)
        worse_list.append(worse)

    # average over 20 trials
    accept_list = np.array(accept_list)         # used to plot acceptance probability
    worse_list = np.array(worse_list).mean(0)       # take maximum, used in tables

    # print((true_return_list))

    return worse_list.max(), accept_list, worse_list, n_eps

def brex_safety_test(env_name, epsilon, n_eps = None):
    """
    B-REX style safety test
    """

    accept_list = []
    worse_list = []

    for s in range(1, 21):
        with open(os.path.join("evaluation", env_name.replace("Vision", ""), "demo", f"returns_seed{s}.pkl"), "rb") as f:
            data = pickle.load(f)

        demo_returns = data['st_returns'].mean(0).reshape(1, -1) * epsilon
        demo_true_return = data['true_return'].mean() * epsilon

        # print(demo_returns.shape)   # (1, 400)
        # print(demo_true_return)

        # load postpi returns
        try:
            path = os.path.join("evaluation", env_name, "dist", f"epsilon{epsilon}")
            with open(os.path.join(path, f"returns_seed{s}.pkl"), "rb") as f:
                data = pickle.load(f)
        except:
            print("Cannot load data", env_name, epsilon, s)
            continue

        # expected return of postpi policy under ground-truth reward function
        # for trial $s of epsilon $epsilon
        true_return = data['true_return'].mean()

        # in evaluate_model.py, typo: cp_returns instead of st_returns
        # but reward samples and values are correct, just the key name wrong
        returns = data['cp_returns']
        # print(returns.shape)     # (n_eval_episodes, n_rewards = 400)

        return_difference = (returns - demo_returns).T

        # test different number of episodes used for safey test
        if n_eps is None:
            n_eps = np.linspace(2, min(returns.shape[0], 1000), 20)
            n_eps = [int(n) for n in n_eps]

        accept = []     # 1 if policy is accepted
        worse = []      # 1 if accepted policy is worse than initial policy
        for n in n_eps:

            L = np.quantile(return_difference[:, :n].mean(1), args.delta, method="lower")

            if L >= 0:
                # accept
                accept.append(1)
                worse.append(int(true_return < demo_true_return))
            else:
                # reject
                accept.append(0)
                worse.append(0)     # rejected, so safe

        accept_list.append(accept)
        worse_list.append(worse)

    # average over 20 trials
    accept_list = np.array(accept_list)         # used to plot acceptance probability
    worse_list = np.array(worse_list).mean(0)       # take maximum, used in tables

    return worse_list.max(), accept_list, worse_list, n_eps



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--delta", type=float, default=0.05, help="Desired confidence level is (1 - delta).")
    parser.add_argument("--st_comparison", action="store_true")
    parser.add_argument("--cp_comparison", action="store_true")
    args = parser.parse_args()

    colors = ["red", "orange", "green", "blue", "magenta"]
    markers = ["o",  "^", "s", "D", "p"]

    if args.cp_comparison:
        for i, eps in enumerate([0, 0.25, 0.5, 0.75, 1]):
            print("POSTPI", eps, end=": ")
            worse, accept_list, _, _ = postpi_safety_test(args.env_name, eps, n_eps=[1000])
            pg_worse , pg_accept_list, _, _ = baseline_safety_test(args.env_name, eps, alg="pgbroil", n_eps=[1000])
            cpl_worse , cpl_accept_list, _, _ = baseline_safety_test(args.env_name, eps, alg="cpl", n_eps=[1000])


            print(worse, accept_list.mean(0))
            print(pg_worse, pg_accept_list.mean(0))
            print(cpl_worse, cpl_accept_list.mean(0))

        quit()

    if args.st_comparison:

        for i, eps in enumerate([0, 0.25, 0.5, 0.75, 1]):
            print("POSTPI", eps, end=": ")
            worse , accept_list, wl, n_eps = postpi_safety_test(args.env_name, eps, n_eps=[20])
            brex_worse, brex_accept_list, bwl, _ = brex_safety_test(args.env_name, eps, n_eps=[20])

            print(wl)
            # print(accept_list.mean(0))
            print(bwl)
            # print(brex_accept_list.mean(0))

        quit()

    if not "Vision" in args.env_name:

        epsilons = [0, 0.25, 0.5, 0.75, 1]
        modes = ["pgbroil", "mean", "map", "trex", "cpl"]

        true_returns = []
        for mode in modes:
            true_returns.append(load_true_return(args.env_name, mode))

        demo_returns = load_true_return(args.env_name, "demo")
        demo_returns_eps = np.tile(demo_returns, (5, 1)) * np.array(epsilons).reshape(5, 1)
        
        ### Probability of returning policies worse than intial policies
        ### For BASELINES!!
        for mode, ret in zip(modes, true_returns):
            print(mode, end=": ")

            mean =  (ret < demo_returns_eps).astype(int).mean(1)
            print("Mean", mean, end="    ")
        
        ### Probability of returning policies worse than intial policies
        ### For POSTPI
        plt.figure(dpi=300)
        plt.rcParams.update({'font.size': 15}) 
        for i, eps in enumerate([0, 0.25, 0.5, 0.75, 1]):
            print("POSTPI", eps, end=": ")
            worse , accept_list, wl, n_eps = postpi_safety_test(args.env_name, eps)
            print(worse, accept_list.mean(0))

            color = colors[i]
            marker = markers[i]
            accept_mean = accept_list.mean(0)
            # accept_std = accept_list.std(0)
            accept_sterror = accept_list.std(0, ddof=1) / np.sqrt(accept_list.shape[0])
            plt.plot(n_eps, accept_mean, label=f"ε={eps}", color=color, marker=marker)
            plt.fill_between(n_eps, accept_mean - accept_sterror, accept_mean + accept_sterror, color=color, alpha=0.1)

        # plt.legend(loc='lower right')#, fontsize="small")
        plt.ylabel("Probablity of Returning a Policy")
        plt.xlabel("Number of Episodes for High Confidence Bounds")
        plt.savefig(f"plots/prob_accept_{args.env_name}.png", bbox_inches='tight')

    else:
        # Vision experiments
        ### Probability of returning policies worse than intial policies
        ### For POSTPI
        for eps in [0, 0.25, 0.5, 0.75, 1]:
            print("POSTPI", eps, end=": ")
            worse , accept_list, _, _ = postpi_safety_test(args.env_name, eps)
            print(worse, accept_list.mean(0))
