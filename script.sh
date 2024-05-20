#!/bin/bash

####### Goal Domain

##### 1. train demonstration policy
# we found that the learning rate of 1e-4 performed best
# we trained 5 policies (seed 0-4) using lr 1e-4, and found seed 1 to be (among) the best
python train_demo_policy.py SafetyPointGoal5-v0 --total_timesteps 5000000 --save_freq 1000000 --seed 1 --lr 1e-4

##### 2. generate trajectories using trained demonstration policy
python generate_demonstration.py SafetyPointGoal5-v0 --n_demo 30 --return_thres 15

##### 3. Sample pairs of trajectories to form the preference dataset
# repeat for 20 trials
for i in {1..20};
do
    python generate_preferences.py SafetyPointGoal5-v0  --seed $i
done

##### 4. Generate reward samples using preferences
for i in {1..20};
do
    python generate_reward_samples.py SafetyPointGoal5-v0 --proposal_width 0.1 --seed $i
done

##### 5. Run POSTPI
### (a)
# evaluate the performance of the initial (demo) policy under reward samples
# repeat for 20 trials
for i in {1..20};
do
    python evaluate_model.py SafetyPointGoal5-v0 demo candidate_selection/SafetyPointGoal5-v0/true-0.0001-1/rl_model_seed1_5000000_steps.zip --seed $i
done

### (b)
# run candidate proposal, for all epsilons
for eps in 0 0.25 0.5 0.75 1;
do
        for i in {1..20};
        do
                python candidate_selection.py SafetyPointGoal5-v0 --exp_name epsilon$eps --epsilon $eps --total_timesteps 5000000 --save_freq 1000000 --mode dist --alpha 0.975 --trial_seed $i
        done
done

##### 6. Baselines

### (a) PG-BROIL
for i in {1..20};
do
    python candidate_selection.py SafetyPointGoal5-v0 --mode pgbroil --total_timesteps 5000000 --save_freq 1000000 --alpha 0.95 --trial_seed $i
done

### (b) B-REX Mean
for i in {1..20};
do
    python candidate_selection.py SafetyPointGoal5-v0 --mode mean --total_timesteps 5000000 --save_freq 1000000 --trial_seed $i
done

### (c) B-REX MAP
for i in {1..20};
do
    python candidate_selection.py SafetyPointGoal5-v0 --mode map --total_timesteps 5000000 --save_freq 1000000 --trial_seed $i
done


### (d) T-REX
# (i) Generate T-REX reward
for i in {1..20}:
do
    python generate_trex_reward.py SafetyPointGoal5-v0 --seed $SLURM_ARRAY_TASK_ID
done

# (ii) Run optimization
for i in {1..20};
do
    python candidate_selection.py SafetyPointGoal5-v0 --mode trex --total_timesteps 5000000 --save_freq 1000000  --trial_seed $i
done

### (e) CPL
# (i) Generate CPL preferences
for i in {1..20};
do
    python generate_cpl_preferences.py SafetyPointGoal5-v0 candidate_selection/SafetyPointGoal5-v0/true-0.0001-1/rl_model_seed1_5000000_steps.zip --seed $i
done

# (ii) Run CPL
for i in {1..20};
do
    python train_cpl.py SafetyPointGoal5-v0 --seed $i
done

##### 7. Evaluate trained models
### (a) POSTPI
for eps in 0 0.25 0.5 0.75 1;
do
        for i in {1..20};
        do
                python evaluate_model.py SafetyPointGoal5-v0 dist candidate_selection/SafetyPointGoal5-v0/dist/epsilon$eps --seed  $i
        done
done

### (b) PG-BROIL
for i in {1..20};
do
    python evaluate_model.py SafetyPointGoal5-v0 pgbroil candidate_selection/SafetyPointGoal5-v0/pgbroil/ --seed $i
done

### (c) B-REX Mean
for i in {1..20};
do
    python evaluate_model.py SafetyPointGoal5-v0 mean candidate_selection/SafetyPointGoal5-v0/mean/ --seed $i
done

### (d) B-REX MAP
for i in {1..20};
do
    python evaluate_model.py SafetyPointGoal5-v0 map candidate_selection/SafetyPointGoal5-v0/map/ --seed $i
done

### (e) T-REX
for i in {1..20};
do
    python evaluate_model.py SafetyPointGoal5-v0 trex candidate_selection/SafetyPointGoal5-v0/trex/ --seed $i
done

### (f) CPL
for i in {1..20};
do
    python evaluate_model.py SafetyPointGoal5-v0 cpl candidate_selection/SafetyPointGoal5-v0/cpl/ --seed $i
done

##### 8. Perform safety test & Generate Results
python safety_test.py SafetyPointGoal5-v0 

####### Circle Domain

##### 1. train demonstration policy
# we found that the learning rate of 1e-4 performed best
# we trained 5 policies (seed 0-4) using lr 1e-4, and found seed 0 to be (among) the best
python train_demo_policy.py SafetyPointCircle1-v0 --total_timesteps 3000000 --save_freq 1000000 --seed 0 --lr 1e-4

##### 2. generate trajectories using trained demonstration policy
python generate_demonstration.py SafetyPointCircle1-v0 --n_demo 30 --return_thres 30

##### 3. Sample pairs of trajectories to form the preference dataset
# repeat for 20 trials
for i in {1..20};
do
    python generate_preferences.py SafetyPointCircle1-v0 --seed $i
done

##### 4. Generate reward samples using preferences
for i in {1..20};
do
    python generate_reward_samples.py SafetyPointCircle1-v0 --seed $i
done

##### 5. POSTPI
### (a)
# evaluate the performance of the initial (demo) policy under reward samples
# repeat for 20 trials
for i in {1..20};
do
    python evaluate_model.py SafetyPointCircle1-v0 demo candidate_selection/SafetyPointCircle1-v0/true-0.0001-0/rl_model_seed0_3000000_steps.zip --seed $i
done

### (b)
# run candidate proposal, for all epsilons
for eps in 0 0.25 0.5 0.75 1;
do
        for i in {1..20};
        do
                python candidate_selection.py SafetyPointCircle1-v0 --exp_name epsilon$eps --epsilon $eps --total_timesteps 3000000 --save_freq 1000000 --mode dist --alpha 0.975 --trial_seed $i
        done
done

##### 6. Baselines

### (a) PG-BROIL
for i in {1..20};
do
    python candidate_selection.py SafetyPointCircle1-v0 --mode pgbroil --total_timesteps 3000000 --save_freq 1000000 --alpha 0.95 --trial_seed $i
done

### (b) B-REX Mean
for i in {1..20};
do
    python candidate_selection.py SafetyPointCircle1-v0 --mode mean --total_timesteps 3000000 --save_freq 1000000 --trial_seed $i
done

### (c) B-REX MAP
for i in {1..20};
do
    python candidate_selection.py SafetyPointCircle1-v0 --mode map --total_timesteps 3000000 --save_freq 1000000 --trial_seed $i
done

### (d) T-REX
# (i)
# Generate T-REX reward
for i in {1..20}:
do
    python generate_trex_reward.py SafetyPointCircle1-v0 --seed $SLURM_ARRAY_TASK_ID
done

# (ii)
# Run optimization
for i in {1..20};
do
    python candidate_selection.py SafetyPointCircle1-v0 --mode trex --total_timesteps 3000000 --save_freq 1000000  --trial_seed $i
done

### (e) CPL
# (i) Generate CPL preferences
for i in {1..20};
do 
    python generate_cpl_preferences.py SafetyPointCircle1-v0 candidate_selection/SafetyPointCircle1-v0/true-0.0001-0/rl_model_seed0_3000000_steps.zip --seed $i
done

# (ii) Run CPL
for i in {1..20};
do
    python train_cpl.py SafetyPointCircle1-v0 --seed $i
done

##### 7. Evaluate trained models
### (a) POSTPI
for eps in 0 0.25 0.5 0.75 1;
do
        for i in {1..20};
        do
                python evaluate_model.py SafetyPointCircle1-v0 dist candidate_selection/SafetyPointCircle1-v0/dist/epsilon$eps --seed $i
        done
done

### (b) PG-BROIL
for i in {1..20};
do
    python evaluate_model.py SafetyPointCircle1-v0 pgbroil candidate_selection/SafetyPointCircle1-v0/pgbroil/ --seed $i
done

### (c) B-REX Mean
for i in {1..20};
do
    python evaluate_model.py SafetyPointCircle1-v0 mean candidate_selection/SafetyPointCircle1-v0/mean/ --seed $i
done

### (d) B-REX MAP
for i in {1..20};
do
    python evaluate_model.py SafetyPointCircle1-v0 map candidate_selection/SafetyPointCircle1-v0/map/ --seed $i
done

### (e) T-REX
for i in {1..20};
do
    python evaluate_model.py SafetyPointCircle1-v0 trex candidate_selection/SafetyPointCircle1-v0/trex/ --seed $i
done

### (f) CPL
for i in {1..20};
do
    python evaluate_model.py SafetyPointCircle1-v0 cpl candidate_selection/SafetyPointCircle1-v0/cpl/ --seed $i
done

##### 8. Perform safety test & Generate Results
python safety_test.py SafetyPointCircle1-v0 