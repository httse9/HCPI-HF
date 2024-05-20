import safety_gymnasium as gym
import gymnasium
import torch
import os
import numpy as np
import pickle

"""
Modes:
dist: postpi
pgbroil: pgbroil
mean: b-rex mean
map: b-rex map
trex: t-rex
true: optimize wrt to ground-truth reward

modes are named this way for historical reasons...
"""

class SafetyGoalFeatureWrapper(gymnasium.Wrapper):
    """
    Hand design reward features
    """
    def __init__(self, env, env_name, samples_filename=None, mode="dist", samples_type=["cp"]):
        super().__init__(env)
        self.true_reward = self.get_true_reward()
        # self.task_id = env_name
        self.env_name = env_name
        if "Vision" in self.env_name:
            self.observation_space = self.observation_space['vision']

        self.samples_filename = samples_filename
        self.mode = mode

        if mode == "true":
            self.reward_samples = self.true_reward.reshape(1, -1)


        if samples_filename is not None:
            # load reward samples file
            samples_path = os.path.join("./reward_samples", env_name.replace("Vision", ""), samples_filename)
            with open(samples_path, "rb") as f:
                samples = pickle.load(f)

            if mode == "dist" or mode == "pgbroil":
                # for postpi or pgbroil, load the entire set of reward samples (candidate proposal set)

                reward_samples = []
                # we have two sets of reward samples, one reserved for candidate proposal
                # and one reserved for safety test. samples_type determines
                # which (or both) set to use.
                if "cp" in samples_type:
                    reward_samples.append(samples['samples'])
                if "st" in samples_type:
                    reward_samples.append(samples['test_samples'])
                
                self.reward_samples = np.concatenate(reward_samples)

            elif mode == "mean":
                # load the mean reward
                self.reward_samples = samples['mean_r'].reshape(1, -1)

            elif mode == "map":
                # load the map reward
                self.reward_samples = samples['map_r'].reshape(1, -1)

            elif mode == "trex":
                # load the t-rex reward
                self.reward_samples = samples.reshape(1, -1)
            

    def get_true_reward(self):
        # Get the ground-truth reward R*
        if "Goal" in self.task_id:
            # [dense reward, in goal, in hazard1, ..., in hazard n]
            true_reward = [1, 1]
            true_reward += [-1] * self.env.task._obstacles[1].num
            return np.array(true_reward).astype(float)
        elif "Circle" in self.task_id:
            # [dense reward, in circle, out of boundary]
            # return np.array([1, 0, -1]).astype(float)

            # [dense reward, out of boundary]
            return np.array([1, -1]).astype(float)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if "Goal" in self.task_id:
            info['state_features'] = np.array([0.0] * 6)
        elif "Circle" in self.task_id:
            info['state_features'] = np.array([0.0] * 2)

        if "Vision" in self.env_name:
            return obs['vision'], info
        else:
            return obs, info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        if "Vision" in self.env_name:
            obs = obs['vision']
        
        # construct state features
        features = self.state_features(reward, cost)
        info["state_features"] = features

        # if self.samples_filename is not None or self.mode == "true":
        info["rewards"] = self.get_rewards(features)
        return obs, reward - cost, terminated, truncated, info


    def get_rewards(self, features):
        return self.reward_samples @ features

    def get_n_rewards(self):
        # return number of rewrad samples
        if self.mode in ["mean", "true", "map", "trex"]:
            return 1
        elif self.mode == "dist" or self.mode == "pgbroil":
            return self.reward_samples.shape[0]

    def state_features(self, reward, cost):
        if "Goal" in self.task_id:
            return self.goal_state_features(reward)
        elif "Circle" in self.task_id:
            return self.circle_state_features(reward, cost)

    def circle_state_features(self, reward, cost):
        # [dense reward, agent out of bound]
        features = [reward]
        features += [cost]

        return np.array(features)

    def goal_state_features(self, reward):
        """
        reward: actual reward at this time step,
        used for calculating dense reward component
        """
        # [dense reward, agent in goal, agent in hazard 1, ..., agent in hazard n]
        features = []

        goal_achieved = int(np.round(reward))

        # feature: dense reward
        features.append(reward - goal_achieved)
        # feature: agent in goal
        features.append(goal_achieved)
        
        # feature: agent in hazard
        hazards = self.env.task._obstacles[1]
        for h_pos in hazards.pos:
            h_dist = hazards.agent.dist_xy(h_pos)
            violate = int(h_dist <= hazards.size)
            features.append(violate)

        return np.array(features)

        