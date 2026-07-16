# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    # Prioritized Experience Replay arguments
    per_alpha: float = 0.6
    """priority exponent; 0 means uniform replay"""

    per_beta_start: float = 0.4
    """initial importance-sampling correction exponent"""

    per_beta_end: float = 1.0
    """final importance-sampling correction exponent"""

    per_eps: float = 1e-6
    """small positive constant added to absolute TD errors"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

@dataclass
class PrioritizedReplayBatch:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay buffer."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        alpha: float,
        eps: float,
    ):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be greater than 0")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("only discrete action spaces are supported")

        self.buffer_size = buffer_size
        self.device = device
        self.alpha = alpha
        self.eps = eps

        observation_shape = observation_space.shape

        self.observations = np.zeros(
            (buffer_size, *observation_shape),
            dtype=np.float32,
        )
        self.next_observations = np.zeros(
            (buffer_size, *observation_shape),
            dtype=np.float32,
        )
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.priorities = np.zeros(buffer_size, dtype=np.float32)

        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def add(
        self,
        observation,
        next_observation,
        action,
        reward,
        done,
        infos=None,
    ):
        index = self.position

        self.observations[index] = np.asarray(
            observation[0],
            dtype=np.float32,
        )
        self.next_observations[index] = np.asarray(
            next_observation[0],
            dtype=np.float32,
        )
        self.actions[index, 0] = int(np.asarray(action).reshape(-1)[0])
        self.rewards[index, 0] = float(np.asarray(reward).reshape(-1)[0])
        self.dones[index, 0] = float(np.asarray(done).reshape(-1)[0])

        # New transitions receive the current maximum priority.
        self.priorities[index] = self.max_priority

        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int, beta: float):
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        active_priorities = self.priorities[: self.size]
        scaled_priorities = active_priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(
            self.size,
            size=batch_size,
            replace=True,
            p=probabilities,
        )

        weights = (self.size * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        return PrioritizedReplayBatch(
            observations=torch.as_tensor(
                self.observations[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            actions=torch.as_tensor(
                self.actions[indices],
                dtype=torch.long,
                device=self.device,
            ),
            rewards=torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            dones=torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            weights=torch.as_tensor(
                weights,
                dtype=torch.float32,
                device=self.device,
            ),
            indices=indices,
        )

    def update_priorities(self, indices, priorities):
        if torch.is_tensor(priorities):
            priorities = priorities.detach().cpu().numpy()

        priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)

        for index, priority in zip(indices, priorities):
            updated_priority = max(abs(float(priority)), self.eps)
            self.priorities[int(index)] = updated_priority
            self.max_priority = max(self.max_priority, updated_priority)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = PrioritizedReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        alpha=args.per_alpha,
        eps=args.per_eps,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                # Anneal beta from per_beta_start to per_beta_end.
                beta_progress = min(1.0, global_step / args.total_timesteps)
                beta = args.per_beta_start + beta_progress * (
                        args.per_beta_end - args.per_beta_start
                )

                data = rb.sample(args.batch_size, beta=beta)

                with torch.no_grad():
                    target_max, _ = target_network(
                        data.next_observations
                    ).max(dim=1)

                    td_target = (
                            data.rewards.flatten()
                            + args.gamma
                            * target_max
                            * (1 - data.dones.flatten())
                    )

                old_val = (
                    q_network(data.observations)
                    .gather(1, data.actions)
                    .squeeze(1)
                )

                td_error = td_target - old_val

                # Importance-sampling weighted TD loss.
                loss = (
                        data.weights * td_error.pow(2)
                ).mean()

                # Updated TD errors determine future sampling priorities.
                rb.update_priorities(
                    data.indices,
                    td_error.detach().abs(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
