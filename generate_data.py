import safety_gymnasium as gym
from stable_baselines3 import PPO
import os
import numpy as np
import pickle

def generate_trajectory(env, model, state_features, verbose=True):

    obs, _ = env.reset()

    ret = 0
    obs_list = []
    act_list = []
    rew_list = []

    done = False
    while not done:

        a, _ = model.predict(obs, deterministic=True)
        
        obs, r, cost, terminated, truncated, info = env.step(a)
        r = r - cost

        act_list.append(a)
        ret += r
        rew_list.append(r)

        if state_features:
            obs_list.append(info["state_features"])
        else:
            obs_list.append(obs)

        if terminated or truncated:
            break

    observations = np.array(obs_list)
    actions = np.array(act_list)
    rewards = np.array(rew_list)
    
    # print(observations.shape)
    # print(actions.shape)
    # print(ret)

    if verbose:
        print("  Trajectory generated with return:", ret)

    return (observations, actions, rewards), ret

def filter_checkpoints(checkpoints, min_t=None, max_t=None):
    # filter checkpoints based on checkpoint timestep
    if min_t is None and max_t is None:
        return checkpoints

    filtered = []
    for cp in checkpoints:
        timestep = int(cp.split("_")[3])

        if min_t is not None and timestep < min_t:
            continue
        if max_t is not None and timestep > max_t:
            continue

        filtered.append(cp)
    return filtered


def generate_demo(args, verbose=True):
    # create env
    env = gym.make(args.env_name)
    if args.state_feature_available:
        from env_wrapper import SafetyGoalFeatureWrapper
        env = SafetyGoalFeatureWrapper(env, args.env_name)

    # retrieve checkpoints
    checkpoint_path = os.path.join("./candidate_selection", args.env_name, "true")
    checkpoints = os.listdir(checkpoint_path)
    checkpoints = [cp for cp in checkpoints if "zip" in cp]
    checkpoints = filter_checkpoints(checkpoints, min_t=args.min_t, max_t = args.max_t)

    if verbose:
        print(f"Found {len(checkpoints)} checkpoints in '{checkpoint_path}' .")

    demos = []
    returns = []

    n_checkpoints = min(args.n_checkpoints, len(checkpoints))
    n_checkpoints = min(args.n_demo, n_checkpoints)
    n_demo_per_checkpoint = int(np.ceil(args.n_demo / n_checkpoints))

    if verbose:
        print(f"Generating {n_demo_per_checkpoint} trajectories for each of {n_checkpoints} checkpoints.")

    random_checkpoints = np.random.choice(checkpoints, size=n_checkpoints, replace=False)

    for checkpoint in random_checkpoints:
        
        path = os.path.join(checkpoint_path, checkpoint)
        model = PPO.load(path)
        if verbose:
            print("Picked checkpoint", checkpoint)

        # generate trajectories using picked checkpoint
        if "Circle" in args.env_name:
            count = 0
            while count < n_demo_per_checkpoint:
                demo, ret = generate_trajectory(env, model, args.state_feature_available)

                if args.return_thres is not None:
                    if ret >= args.return_thres:
                        demos.append(demo)
                        returns.append(ret)
                        count += 1
                else:
                    demos.append(demo)
                    returns.append(ret)
                    count += 1

        elif "Goal" in args.env_name:
            for _ in range(n_demo_per_checkpoint):
                demo, ret = generate_trajectory(env, model, args.state_feature_available)

                if args.return_thres is not None:
                    if ret >= args.return_thres:
                        demos.append(demo)
                        returns.append(ret)
                else:
                    demos.append(demo)
                    returns.append(ret)

    return demos, returns

def generate_random_index_pair(n_demo):
    i = np.random.randint(0, n_demo)
    j = np.random.randint(0, n_demo)

    while (i == j):
        j = np.random.randint(0, n_demo)

    return i, j
    

def generate_reward_model_training_data(args, verbose=True):
    demos, returns = generate_demo(args)

    if verbose:
        print("Generating training data for reward model.")

    training_obs = []       # pairs of trajectories
    training_act = []
    training_reward = []              
    training_label = []

    training_trajs = [d[0] for d in demos]     # all trajectories
    training_rew = [d[2] for d in demos]       # all rewards

    wrong_label_count = 0

    # generate n_demo pairs of trajectories
    # according to Bradley Terry model
    for _ in range(args.n_demo):
        i, j = generate_random_index_pair(len(returns))

        # bradley terry model
        prob_i_better_than_j = 1 / (1 + np.exp(args.beta * (returns[j] - returns[i])))

        if np.random.uniform() < prob_i_better_than_j:
            label = 0

            if returns[j] > returns[i]:
                wrong_label_count += 1
        else:
            label = 1

            if returns[i] > returns[j]:
                wrong_label_count += 1

        training_obs.append([demos[i][0], demos[j][0]])
        training_label.append(label)
        training_act.append([demos[i][1], demos[j][1]])
        training_reward.append([demos[i][2], demos[j][2]])

    if verbose:
        print("Wrong labels:", wrong_label_count)

    training_obs = np.array(training_obs)
    training_act = np.array(training_act)
    training_reward = np.array(training_reward)
    training_label = np.array(training_label)
    training_trajs = np.array(training_trajs)
    training_rew = np.array(training_rew)

    # print(training_obs.shape, training_act.shape, training_label.shape)
    # (n_demo, 2, n_steps, obs_dim), (n_demo, 2, n_steps, act_dim), (n_demo)

    # print(training_rew.shape)     # (n_trajs, n_steps)
    # print(training_trajs.shape)     # (n_trajs, n_steps, obs_dim)

    # save reward model training data
    reward_model_data = {
        "obs": training_obs,
        "act": training_act,
        "reward": training_reward,
        "label": training_label,
        "traj": training_trajs,
        "rew": training_rew
    }
    save_path = os.path.join("./reward_models", args.env_name)
    os.makedirs(save_path, exist_ok=True)

    if args.state_feature_available:
        filename = "reward_model_data_stateFeatAvail.pkl"
    else:
        filename = "reward_model_data.pkl"

    with open(os.path.join(save_path, filename), "wb") as f:
        pickle.dump(reward_model_data, f)

    # return training_obs, training_act, training_label, training_rew

    if verbose:
        print("Finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Environment Name")
    parser.add_argument("--n_checkpoints", type=int, default=10, help="Number of checkpoints used to generate trajectories")
    parser.add_argument("--n_demo", type=int, default=10, help="Number of demonstration (pairs of trajectories)")
    parser.add_argument("--state_feature_available", action="store_true", help="Whether to use pre-designed state features.")
    parser.add_argument("--beta", type=float, default=1, help="Inverse temperature paramter in Bradley-Terry model.")
    parser.add_argument("--min_t", type=int, default=None, help="Minimum timestep of checkpoints to use.")
    parser.add_argument("--max_t", type=int, default=100000, help="Maximum timestep of checkpoints to use.")
    parser.add_argument("--return_thres", default=None, type=float, help="Only use episodes with return above this threshold.")
    args = parser.parse_args()

    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(1)

    generate_reward_model_training_data(args)