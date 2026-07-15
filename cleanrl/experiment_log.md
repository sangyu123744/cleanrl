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
···
### Result

- Final episodic return: 500
- Final episodic length: 500
- CartPole-v1 maximum return reached
- Training completed without errors
- TensorBoard log directory: `runs/CartPole-v1__ppo__1__1784032010`

### Conclusion

The standard PPO implementation from CleanRL was successfully reproduced on CartPole-v1. The agent reached the maximum episodic return of 500.