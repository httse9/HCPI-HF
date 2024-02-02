# High-Confidence Policy Improvement from Human Feedback
---
## Installation:
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
pip install -e .[docs,tests,extra]
```
4. Install packages.
```
cd ..
pip install -r requirements.txt
```
---
## Usage:
Refer to script.sh for examples of how to run the code.
