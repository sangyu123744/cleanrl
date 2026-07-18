# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
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

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Prioritized trajectory replay arguments
    replay_buffer_size: int = 32
    """maximum number of rollout batches stored in the replay buffer"""

    replay_ratio: float = 0.5
    """fraction of optimization data sampled from replayed rollouts"""

    per_alpha: float = 0.6
    """priority exponent; 0 means uniform sampling"""

    per_beta: float = 0.4
    """importance-sampling correction exponent"""

    per_eps: float = 1e-6
    """small constant preventing zero priority"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class PrioritizedRolloutBuffer:
    """Stores old PPO rollout batches and samples them by priority."""

    def __init__(self, capacity: int, alpha: float, eps: float):
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be greater than 0")

        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.storage = []
        self.priorities = []

    def __len__(self):
        return len(self.storage)

    def add(self, rollout: dict, priority: float):
        """Add one rollout batch and copy tensors to CPU memory."""
        stored_rollout = {
            key: value.detach().cpu().clone()
            for key, value in rollout.items()
        }

        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
            self.priorities.pop(0)

        self.storage.append(stored_rollout)
        self.priorities.append(max(float(priority), self.eps))

    def sample(self, sample_size: int, beta: float):
        """Sample rollout batches and return importance-sampling weights."""
        if len(self.storage) == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        sample_size = min(sample_size, len(self.storage))

        priorities = np.asarray(self.priorities, dtype=np.float64)
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            len(self.storage),
            size=sample_size,
            replace=False,
            p=probabilities,
        )

        weights = (len(self.storage) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        sampled_rollouts = [self.storage[index] for index in indices]
        weights = torch.tensor(weights, dtype=torch.float32)

        return sampled_rollouts, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities after replayed data has been optimized."""
        for index, priority in zip(indices, priorities):
            self.priorities[int(index)] = max(float(priority), self.eps)

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
def compute_ppo_losses(
    agent,
    observations,
    actions,
    old_logprobs,
    advantages,
    returns,
    old_values,
    clip_coef,
    clip_vloss,
    norm_adv,
    sample_weights=None,
):
    """Compute PPO losses for one minibatch."""

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
        observations,
        actions.long(),
    )

    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = (
            (ratio - 1.0).abs() > clip_coef
        ).float().mean()

    if norm_adv:
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

    def weighted_mean(values):
        values = values.reshape(-1)

        if sample_weights is None:
            return values.mean()

        weights = sample_weights.to(
            device=values.device,
            dtype=values.dtype,
        ).reshape(-1)

        if weights.numel() != values.numel():
            raise ValueError(
                "sample_weights and loss values must have the same size"
            )

        return (weights * values).mean()

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(
        ratio,
        1 - clip_coef,
        1 + clip_coef,
    )
    pg_loss = weighted_mean(
        torch.max(pg_loss1, pg_loss2)
    )
    # Value loss
    newvalue = newvalue.view(-1)

    if clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2

        v_clipped = old_values + torch.clamp(
            newvalue - old_values,
            -clip_coef,
            clip_coef,
        )

        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_per_sample = 0.5 * torch.max(
            v_loss_unclipped,
            v_loss_clipped,
        )
    else:
        v_loss_per_sample = 0.5 * (
            (newvalue - returns) ** 2
        )

    v_loss = weighted_mean(v_loss_per_sample)
    entropy_loss = weighted_mean(entropy)

    return (
        pg_loss,
        v_loss,
        entropy_loss,
        old_approx_kl,
        approx_kl,
        clipfrac,
    )
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    replay_buffer = PrioritizedRolloutBuffer(
        capacity=args.replay_buffer_size,
        alpha=args.per_alpha,
        eps=args.per_eps,
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)



        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    old_approx_kl,
                    approx_kl,
                    clipfrac,
                ) = compute_ppo_losses(
                    agent=agent,
                    observations=b_obs[mb_inds],
                    actions=b_actions[mb_inds],
                    old_logprobs=b_logprobs[mb_inds],
                    advantages=b_advantages[mb_inds],
                    returns=b_returns[mb_inds],
                    old_values=b_values[mb_inds],
                    clip_coef=args.clip_coef,
                    clip_vloss=args.clip_vloss,
                    norm_adv=args.norm_adv,
                )

                clipfracs.append(clipfrac.item())
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
            # Store the completed on-policy rollout for future replay.
            current_rollout = {
                "obs": b_obs,
                "actions": b_actions,
                "logprobs": b_logprobs,
                "advantages": b_advantages,
                "returns": b_returns,
                "values": b_values,
            }

            initial_priority = (
                    b_returns - b_values
            ).abs().mean().item()

            replay_buffer.add(
                current_rollout,
                initial_priority,
            )

            writer.add_scalar(
                "replay/buffer_size",
                len(replay_buffer),
                global_step,
            )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
