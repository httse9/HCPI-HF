# High-Confidence Policy Improvement from Human Feedback
---
## Requirements:
1. Create conda environment.
```
conda create -n hcpihf python=3.9
``` 
2. Install safety-gymnasium.
```
conda activate hcpihf
cd safety-gymnasium
pip install -e .
```
3. Install stable-baselines3
```
cd ../stable-baselines3
pip install -e .
```
4. Install packages.
```
cd ..
pip install -r requirements.txt
```
---
## Training & Evaluation:
script.sh contains step by step instructions with all commands to generate the main results of the paper.
