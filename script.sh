#!/bin/bash


### Goal domain
# train demonstration policy
python train_demo_policy.py SafetyPointGoal5-v0 --total_timesteps 5000000 --save_freq 1000000

# generate set of preferences
python generate_data.py SafetyPointGoal5-v0 --state_feature_available --min_t 5000000 --max_t 5000000 --beta 5 --n_demo 50 --return_thres 10

# generate reward samples
python bayesian_rex.py SafetyPointGoal5-v0 --beta 5 --proposal_width 0.1 --seed 1

# evaluate initial policy to be used as performance threshold in candidate proposal
# python evaluate_model.py SafetyPointGoal5-v0 ppo optimal candidate_selection/SafetyPointGoal5-v0/true/rl_model_seed0_5000000_steps.zip

# candidate proposal
# can use same python file to run trex/pgbroil/brex by changing the mode argument
# python candidate_selection.py SafetyPointGoal5-v0 --total_timesteps 5000000 --save_freq 1000000 --mode dist --seed 1 --threshold_name optimal --epsilon 1 --exp_name epsilon1

# evaluate candidate policy and generate rollouts for safety test
# python evaluate_model.py SafetyPointGoal5-v0 our thres_init_opt1 candidate_selection/SafetyPointGoal5-v0/dist/epsilon1/ --seed 1 --test --total_timesteps 5000000

# safety test (after running POSTPI and all baselines)
# python safety_test.py SafetyPointGoal5-v0

### Circle domain
# python train_demo_policy.py SafetyPointCircle1-v0 --total_timesteps 3000000 --save_freq 1000000
# python generate_data.py SafetyPointCircle1-v0 --state_feature_available --min_t 3000000 --max_t 3000000 --beta 5 --n_demo 20 --return_thres 35
# python bayesian_rex.py SafetyPointCircle1-v0 --beta 5 --proposal_width 1 --seed 1
# python evaluate_model.py SafetyPointCircle1-v0 ppo optimal candidate_selection/SafetyPointCircle1-v0/true/rl_model_seed0_3000000_steps.zip
# python candidate_selection.py SafetyPointCircle1-v0 --total_timesteps 3000000 --save_freq 1000000 --mode dist --seed 1 --threshold_name optimal --epsilon 1 --exp_name thres_init_opt1
# python evaluate_model.py SafetyPointCircle1-v0 our thres_init_opt1 candidate_selection/SafetyPointCircle1-v0/dist/thres_init_opt1/ --seed 1 --test --total_timesteps 3000000
# python safety_test.py SafetyPointCircle1-v0 

