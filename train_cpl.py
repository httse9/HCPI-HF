import pickle
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import set_random_seed
import time

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

def train_cpl(args, verbose=True):
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # training data
    with open(os.path.join("reward_models", args.env_name, "CPL_data.pkl"), "rb") as f:
        data = pickle.load(f)

    obs = torch.tensor(data["obs"], device=device).float()
    act = torch.tensor(data['act'], device=device).float()
    label = torch.tensor(data['label'], device=device).long()

    model = CPLModel(obs.size(-1), act.size(-1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    bc_losses = []
    cpl_losses = []
    accuracies = []
    losses = [] 

    for e in range(args.n_epochs):

        print(e)

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

    print(bc_losses)
    print(cpl_losses)
    print(accuracies)
    print(losses)

    # save model
    save_path = os.path.join("candidate_selection", args.env_name, "cpl")
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_path, f"model_seed{args.seed}.pt"))

    plt.plot(bc_losses, label="bc")
    plt.plot(cpl_losses, label="cpl")
    plt.plot(accuracies, label="acc")
    plt.plot(losses, label="loss")

    plt.legend()
    plt.savefig(os.path.join(save_path, f"CPL_seed{args.seed}.png"))

    print("Time:", time.time() - start_time)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--n_epochs", default=500, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--bc_steps", default=0, type=int)
    parser.add_argument("--bc_coeff", default=0, type=float)
    parser.add_argument("--lambd", default=0.5, type=float)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    set_random_seed(args.seed)
    train_cpl(args)