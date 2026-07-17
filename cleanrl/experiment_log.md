# Experiment Log

## PPO Baseline — CartPole-v1

- Algorithm: PPO
- Codebase: CleanRL
- Environment: CartPole-v1
- Seed: 1
- Total timesteps: 100000
- Number of environments: 1
- Python: 3.9.23
- PyTorch: 2.5.1
- CUDA available: True

### Command

```bash
python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 100000 --num-envs 1 --seed 1
```

### Result and Conclusion

- Final episodic return: 500
- Final episodic length: 500
- CartPole-v1 maximum return reached
- Training completed without errors
- TensorBoard log directory: `runs/CartPole-v1__ppo__1__1784032010`

The standard PPO implementation from CleanRL was successfully reproduced on CartPole-v1. The agent reached the maximum episodic return of 500.

## PPO Baseline — CartPole-v1 — Seed 2
- Algorithm: PPO
- Codebase: CleanRL
- Environment: CartPole-v1
- Seed: 2
- Total timesteps: 100000
- Number of environments: 1
- Python: 3.9.23
- PyTorch: 2.5.1
- CUDA available: True

### Command

```bash
python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 100000 --num-envs 1 --seed 2
```

### Result and Conclusion

- Maximum observed episodic return: 500
- Training completed without errors
- Late-stage episodic returns showed noticeable fluctuation
- TensorBoard log directory: `runs/CartPole-v1__ppo__2__1784093238`

The CleanRL PPO implementation also reached the maximum CartPole-v1 return under seed 2. This confirms that the successful learning observed with seed 1 was not limited to a single random initialization, although the training return remained variable between episodes.
## DQN Baseline — CartPole-v1 — Seed 1

- Algorithm: DQN
- Codebase: CleanRL
- Environment: CartPole-v1
- Seed: 1
- Total timesteps: 100000

### Command

```bash
python -m cleanrl.dqn --env-id CartPole-v1 --total-timesteps 100000 --seed 1
```

### Result and Conclusion

- Final observed episodic return: 223
- Final smoothed episodic return: approximately 205
- Training completed without errors
- TensorBoard log directory: `runs/CartPole-v1__dqn__1__1784176522`
- CSV result file: `results/CartPole-v1__dqn__1__1784176522.csv`

The DQN baseline learned the CartPole-v1 task under seed 1, but the return did not stably reach the maximum value of 500. Performance declined during the late stage of training.

## DQN Baseline — CartPole-v1 — Seed 2

- Algorithm: DQN
- Codebase: CleanRL
- Environment: CartPole-v1
- Seed: 2
- Total timesteps: 100000

### Command

```bash
python -m cleanrl.dqn --env-id CartPole-v1 --total-timesteps 100000 --seed 2
```

### Result and Conclusion

- Final observed episodic return: 381
- Final smoothed episodic return: approximately 332
- Training completed without errors
- TensorBoard log directory: `runs/CartPole-v1__dqn__2__1784177855`
- CSV result file: `results/CartPole-v1__dqn__2__1784177855.csv`

The DQN baseline achieved better late-stage performance under seed 2 than under seed 1, but it still did not stably reach the maximum return of 500. The difference between the two runs shows that the baseline is sensitive to the random seed.
## DQN+PER — CartPole-v1 — Seed 1

- Algorithm: DQN with Prioritized Experience Replay
- Codebase: CleanRL-based implementation
- Environment: CartPole-v1
- Seed: 1
- Total timesteps: 100000
- PER alpha: 0.6
- PER beta: linearly annealed from 0.4 to 1.0
- PER epsilon: 1e-6

### Command

```bash
python -m cleanrl.dqn_per --env-id CartPole-v1 --total-timesteps 100000 --seed 1
```

### Result and Conclusion

- Final observed episodic return: 451
- Final smoothed episodic return: approximately 443
- Training completed without errors
- TensorBoard log directory: `runs/CartPole-v1__dqn_per__1__1784188987`
- CSV result file: `results/CartPole-v1__dqn_per__1__1784188987.csv`

Under seed 1, DQN with prioritized experience replay achieved substantially higher late-stage performance than the standard DQN baseline. The smoothed episodic return increased from approximately 205 for standard DQN to approximately 443 for DQN+PER.

## DQN+PER — CartPole-v1 — Seed 2

- Algorithm: DQN with Prioritized Experience Replay
- Codebase: CleanRL-based implementation
- Environment: CartPole-v1
- Seed: 2
- Total timesteps: 100000
- PER alpha: 0.6
- PER beta: linearly annealed from 0.4 to 1.0
- PER epsilon: 1e-6

### Command

```bash
python -m cleanrl.dqn_per --env-id CartPole-v1 --total-timesteps 100000 --seed 2
```

### Result and Conclusion

- Final observed episodic return: 207
- Final smoothed episodic return: approximately 208
- Training completed without errors
- TensorBoard log directory: `runs/CartPole-v1__dqn_per__2__1784257827`
- CSV result file: `results/CartPole-v1__dqn_per__2__1784257827.csv`

Under seed 2, DQN+PER performed worse than the standard DQN baseline, whose final smoothed episodic return was approximately 332. The opposite outcomes under seeds 1 and 2 indicate that the current PER implementation and hyperparameter configuration are sensitive to random initialization. Two seeds are insufficient to conclude that PER provides a stable improvement.
## DQN vs DQN+PER — Five-Seed Aggregate Analysis

- Environment: CartPole-v1
- Algorithms: DQN and DQN with Prioritized Experience Replay
- Seeds: 1, 2, 3, 4, 5
- Total timesteps per run: 100000
- Evaluation metric: mean episodic return over the final 20 logged episodes

### Aggregate Results

- DQN final-20 return: 243.55 ± 65.58
- DQN+PER final-20 return: 351.05 ± 88.18
- Absolute improvement: 107.50
- Relative improvement: approximately 44.1%
- Seeds in which DQN+PER outperformed DQN: 4 out of 5
- Mean maximum return:
  - DQN: 475.0
  - DQN+PER: 498.0

Across five random seeds, DQN+PER achieved a substantially higher mean late-stage episodic return than standard DQN. The averaged learning curve also shows that DQN+PER generally maintained higher returns after approximately 25000 environment steps. However, DQN+PER had a larger cross-seed standard deviation and performed worse than standard DQN under seed 2. These results indicate that prioritized experience replay improved overall performance in the current experimental setting, but the method remained sensitive to random initialization.

### Generated Analysis Files

- `results/analysis/per_seed_summary.csv`
- `results/analysis/group_summary.csv`
- `results/analysis/dqn_vs_dqn_per_mean_std.png`