import pickle
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import set_random_seed
import time
import numpy  as np

class CPLModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        ).to(self.device)
        

    def forward(self, obs):
        # return mean action
        return self.policy(obs)

    def predict(self, obs, **kwargs):
        with torch.no_grad():
            action = self.forward(torch.tensor(obs, device=self.device).float())
        return action.cpu().numpy(), None

def train_cpl(args, verbose=True, test=False):
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # training data
    with open(os.path.join("preferences", args.env_name, f"cpl_preferences_seed{args.seed}.pkl"), "rb") as f:
        data = pickle.load(f)

    obs = torch.tensor(data["obs"], device=device).float()
    act = torch.tensor(data['act'], device=device).float()
    label = torch.tensor(data['cpl_label'], device=device).long()

    if test:
        with open(os.path.join("preferences", args.env_name, f"cpl_preferences_seed{args.seed + 1}.pkl"), "rb") as f:
            data = pickle.load(f)

        test_obs = torch.tensor(data["obs"], device=device).float()
        test_act = torch.tensor(data['act'], device=device).float()
        test_label = torch.tensor(data['cpl_label'], device=device).long()

    model = CPLModel(obs.size(-1), act.size(-1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    bc_losses = []
    cpl_losses = []
    accuracies = []
    losses = [] 

    test_bc_losses = []
    test_cpl_losses = []
    test_accuracies = []

    for e in range(args.n_epochs):

        act_mean = model(obs)

        log_prob = - torch.square(act_mean - act).sum(dim=-1)
        # print(log_prob.size())      # (n_pairs, 2, episode_length)

        bc_loss = - log_prob.mean()

        adv = (log_prob * args.alpha).sum(-1)       # (n_pairs, 2)

        # downweight negative sample to favor actions in demo
        for i, l in enumerate(label):
            adv[i, 1 - l] *= args.lambd

        cpl_loss = loss_fn(adv, label)

        with torch.no_grad():
            accuracy = ((adv[:, 1] >= adv[:, 0]).int() == label).float().mean()

        if e < args.bc_steps:
            loss = bc_loss
        else:
            loss = cpl_loss + args.bc_coeff * bc_loss     # bc_coeff = 0.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bc_losses.append(bc_loss.item())
        cpl_losses.append(cpl_loss.item())
        accuracies.append(accuracy.item())
        losses.append(loss.item())

        if test:
            with torch.no_grad():
                act_mean = model(test_obs)
                log_prob = log_prob = - torch.square(act_mean - test_act).sum(dim=-1)

                test_bc_loss = - log_prob.mean()

                adv = (log_prob * args.alpha).sum(-1)       # (n_pairs, 2)
                # downweight negative sample to favor actions in demo
                for i, l in enumerate(test_label):
                    adv[i, 1 - l] *= args.lambd

                test_cpl_loss = loss_fn(adv, test_label)

                test_accuracy = ((adv[:, 1] >= adv[:, 0]).int() == test_label).float().mean()

                test_bc_losses.append(test_bc_loss.item())
                test_cpl_losses.append(test_cpl_loss.item())
                test_accuracies.append(test_accuracy.item())


    # print(bc_losses)
    # print(cpl_losses)
    # print(accuracies)
    # print(losses)

    # save model
    save_path = os.path.join("candidate_selection", args.env_name, "cpl")
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_path, f"model_seed{args.seed}.pt"))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(cpl_losses, label="train")
    plt.plot(test_cpl_losses, label="test")
    plt.legend()
    plt.ylabel("cpl loss")

    plt.subplot(1, 3, 2)
    plt.plot(accuracies, label="train")
    plt.plot(test_accuracies, label="test")
    plt.legend()
    plt.ylabel("acc")

    plt.subplot(1, 3, 3)
    plt.plot(bc_losses, label="train")
    plt.plot(test_bc_losses, label="test")
    plt.legend()
    plt.ylabel("bc loss")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"CPL_seed{args.seed}.png"))

    print("Time:", time.time() - start_time)

    return cpl_losses, bc_losses, accuracies, test_cpl_losses, test_bc_losses, test_accuracies

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--n_epochs", default=1000, type=int)

    parser.add_argument("--alpha", default=0.1, type=float, help="Temp alpha, Table 6 of CPL paper")
    parser.add_argument("--lambd", default=0.5, type=float, help="Bias lambda, Table 6 of CPL paper")
    parser.add_argument("--bc_steps", default=0, type=int, help="Number of steps to pretrain using BC")
    parser.add_argument("--bc_coeff", default=0, type=float, help="BC weight beta, Table 6 of CPL paper")

    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    set_random_seed(42)

    train_cpl(args)

    ### Hyperparameter tuning
    # plt.figure(figsize=(10, 5))
    # for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
    #     args.lr = lr
    #     test_cpl_list = []
    #     test_bc_list = []
    #     test_acc_list = []
    #     for seed in range(1, 6):
    #         args.seed = seed
    #         _, _, _, test_cpl, test_bc, test_acc = train_cpl(args, test=True)
            
    #         test_cpl_list.append(test_cpl)
    #         test_bc_list.append(test_bc)
    #         test_acc_list.append(test_acc)

    #     test_cpl_list = np.array(test_cpl_list).mean(0)
    #     test_bc_list = np.array(test_bc_list).mean(0)
    #     test_acc_list = np.array(test_acc_list).mean(0)

    #     plt.subplot(1, 3, 1)
    #     plt.plot(test_cpl_list, label=lr)
    #     plt.ylabel("cpl loss")

    #     plt.subplot(1, 3, 2)
    #     plt.plot(test_bc_list, label=lr)
    #     plt.ylabel("bc loss")

    #     plt.subplot(1, 3, 3)
    #     plt.plot(test_acc_list, label=lr)
    #     plt.ylabel("acc")
    
    # for i in range(1, 4):
    #     plt.subplot(1, 3, i)
    #     plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(f"CPL_lr.png"))