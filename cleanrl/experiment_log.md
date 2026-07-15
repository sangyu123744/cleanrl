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