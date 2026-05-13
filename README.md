# Deep Reinforcement Learning for Software Quality Assurance

> Test Case Prioritization with DQN and PPO on NASA KC1 & JM1  
> MSc Artificial Intelligence — Deep Learning & Multimedia Data Analysis

---

## Overview

This project applies Deep Reinforcement Learning to the software engineering problem of **Test Case Prioritization (TCP)**. An RL agent learns to identify defect-prone software modules based on static complexity metrics, enabling more efficient allocation of testing resources — finding more defects while running fewer tests.

Two deep RL algorithms are compared on two NASA software defect datasets:

- **DQN (Deep Q-Network):** Learns a Q-function Q(state, action) estimating the expected cumulative reward for testing or skipping each module. Uses experience replay and a target network. Adopts a **precision-focused strategy** — tests fewer modules with higher confidence.
- **PPO (Proximal Policy Optimization):** Learns a policy π(action|state) directly, mapping module metrics to action probabilities. Clips policy updates to ensure stable training. Adopts a **recall-focused strategy** — tests broadly to maximize defect coverage.

Both algorithms are evaluated on **KC1** and **JM1** from the NASA PROMISE Software Engineering Repository — real defect data collected from production software systems.

---

## Results

### KC1 Dataset (2,109 modules · 15.5% defect rate)

| Agent | Recall | Test Rate | Precision | Total Reward |
|---|---|---|---|---|
| Random Baseline | 51.0% | 50.1% | 16.2% | −525 |
| **DQN** | 68.7% | **39.1%** | **27.9%** | −125 |
| **PPO** | **85.1%** | 55.0% | 24.6% | **+138** |

### JM1 Dataset (10,878 modules · 19.4% defect rate)

| Agent | Recall | Test Rate | Precision | Total Reward |
|---|---|---|---|---|
| Random Baseline | 50.0% | 50.0% | 19.3% | −3,193 |
| **DQN** | 62.6% | **21.6%** | **56.1%** | −979 |
| **PPO** | **99.5%** | 96.2% | 20.0% | **+2,046** |

---

## Key Findings

- Both DQN and PPO outperform random testing on both datasets, confirming that software complexity metrics carry learnable signal for defect prediction.
- DQN and PPO converge to **complementary strategies**: DQN prioritizes precision (fewer tests, higher hit rate) while PPO prioritizes recall (broader coverage, fewer missed defects).
- **Dataset size amplifies strategy divergence**: on JM1, DQN tests only 21.6% of modules vs PPO's 96.2% — a gap that is much smaller on the smaller KC1 dataset (39.1% vs 55.0%).
- PPO achieves the highest total reward on both datasets, driven by its recall-focused strategy aligned with the reward function's large penalty for missed defects (−20).
- Training is fast — under 8 minutes per algorithm per dataset on CPU — making the approach viable for periodic retraining as codebases evolve.

---

## Problem Formulation

The task is framed as a Markov Decision Process (MDP):

- **State:** 21 normalized software complexity metrics for the current module
- **Action:** `0` = SKIP, `1` = TEST
- **Reward:**

```
TEST  a defective module  →  +9   (found a bug)
SKIP  a clean module      →   0   (correct skip)
TEST  a clean module      →  −1   (wasted test)
SKIP  a defective module  →  −20  (missed a bug)
```

- **Episode:** one full pass through all modules in the dataset

---

## Datasets

**KC1** — NASA storage management system (C++)
- 2,109 modules · 21 features · 15.5% defect rate
- Split: 1,687 train / 422 test

**JM1** — NASA real-time predictive ground system (C)
- 10,878 modules · 21 features · 19.4% defect rate
- Split: 8,702 train / 2,176 test

Features are 21 Halstead and McCabe software complexity metrics (lines of code, cyclomatic complexity, Halstead volume, difficulty, effort, branch count, and related). All features are normalized to [0, 1] using MinMaxScaler fitted on the training set only.

---

## Algorithm Configuration

| Setting | DQN | PPO |
|---|---|---|
| Total timesteps | 200,000 | 200,000 |
| Network | MLP [64, 64] | MLP [64, 64] |
| Learning rate | 1e-3 | 3e-4 |
| Batch size | 64 | 64 |
| Gamma | 0.95 | 0.95 |
| Replay buffer | 50,000 | — |
| Clip range | — | 0.2 |
| Entropy coefficient | — | 0.01 |
| Training time (KC1) | 3.8 min | 7.4 min |
| Training time (JM1) | 3.8 min | 7.0 min |

---

## Repository Structure

```
├── drl_kc1.ipynb    # KC1 experiment — DQN vs PPO
├── drl_jm1.ipynb    # JM1 experiment — DQN vs PPO
├── DRL_Report.pdf   # Full report
└── README.md
```

---

## Requirements

```bash
stable-baselines3
gymnasium
pandas
numpy
matplotlib
scikit-learn
```

---

## Visualizations

The notebooks include:
- Training reward curves for DQN and PPO
- Defect recall and test rate bar charts
- Summary results table per dataset
