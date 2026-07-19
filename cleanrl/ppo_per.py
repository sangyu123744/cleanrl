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
def build_replay_batch(
    sampled_rollouts,
    rollout_weights,
    sample_count,
    device,
):
    """Merge sampled rollouts and expand rollout weights to sample weights."""

    if len(sampled_rollouts) == 0:
        raise ValueError("sampled_rollouts must not be empty")

    if len(sampled_rollouts) != len(rollout_weights):
        raise ValueError(
            "sampled_rollouts and rollout_weights must have the same size"
        )

    if sample_count <= 0:
        raise ValueError("sample_count must be greater than 0")

    keys = (
        "obs",
        "actions",
        "logprobs",
        "advantages",
        "returns",
        "values",
    )

    merged = {
        key: torch.cat(
            [rollout[key] for rollout in sampled_rollouts],
            dim=0,
        )
        for key in keys
    }

    expanded_weights = torch.cat(
        [
            torch.full(
                (rollout["advantages"].shape[0],),
                float(weight.item()),
                dtype=torch.float32,
            )
            for rollout, weight in zip(
                sampled_rollouts,
                rollout_weights,
            )
        ],
        dim=0,
    )

    total_samples = merged["advantages"].shape[0]
    sample_count = min(sample_count, total_samples)

    selected_indices = torch.randperm(
        total_samples
    )[:sample_count]

    replay_batch = {
        key: value[selected_indices].to(device)
        for key, value in merged.items()
    }

    replay_weights = expanded_weights[
        selected_indices
    ].to(device)

    return replay_batch, replay_weights
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
        # Perform an additional weighted PPO update using historical rollouts.
        replay_sample_count = int(
            args.batch_size * args.replay_ratio
        )

        if len(replay_buffer) > 0 and replay_sample_count > 0:
            rollout_sample_count = min(
                len(replay_buffer),
                max(
                    1,
                    int(
                        np.ceil(
                            replay_sample_count
                            / args.batch_size
                        )
                    ),
                ),
            )

            (
                sampled_rollouts,
                sampled_indices,
                sampled_weights,
            ) = replay_buffer.sample(
                sample_size=rollout_sample_count,
                beta=args.per_beta,
            )

            replay_batch, replay_weights = build_replay_batch(
                sampled_rollouts=sampled_rollouts,
                rollout_weights=sampled_weights,
                sample_count=replay_sample_count,
                device=device,
            )

            replay_size = replay_batch[
                "advantages"
            ].shape[0]

            replay_inds = np.arange(replay_size)
            np.random.shuffle(replay_inds)

            replay_update_count = 0
            replay_last_kl = 0.0

            for start in range(
                    0,
                    replay_size,
                    args.minibatch_size,
            ):
                end = min(
                    start + args.minibatch_size,
                    replay_size,
                )

                replay_mb_inds = torch.as_tensor(
                    replay_inds[start:end],
                    dtype=torch.long,
                    device=device,
                )

                (
                    replay_pg_loss,
                    replay_v_loss,
                    replay_entropy_loss,
                    replay_old_approx_kl,
                    replay_approx_kl,
                    replay_clipfrac,
                ) = compute_ppo_losses(
                    agent=agent,
                    observations=replay_batch["obs"][
                        replay_mb_inds
                    ],
                    actions=replay_batch["actions"][
                        replay_mb_inds
                    ],
                    old_logprobs=replay_batch["logprobs"][
                        replay_mb_inds
                    ],
                    advantages=replay_batch["advantages"][
                        replay_mb_inds
                    ],
                    returns=replay_batch["returns"][
                        replay_mb_inds
                    ],
                    old_values=replay_batch["values"][
                        replay_mb_inds
                    ],
                    clip_coef=args.clip_coef,
                    clip_vloss=args.clip_vloss,
                    norm_adv=args.norm_adv,
                    sample_weights=replay_weights[
                        replay_mb_inds
                    ],
                )

                replay_last_kl = float(
                    replay_approx_kl.item()
                )

                if (
                        args.target_kl is not None
                        and replay_approx_kl
                        > args.target_kl
                ):
                    break

                replay_loss = (
                        replay_pg_loss
                        - args.ent_coef
                        * replay_entropy_loss
                        + args.vf_coef
                        * replay_v_loss
                )

                optimizer.zero_grad()
                replay_loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()

                replay_update_count += 1

            writer.add_scalar(
                "replay/sample_count",
                replay_size,
                global_step,
            )
            writer.add_scalar(
                "replay/rollout_count",
                rollout_sample_count,
                global_step,
            )
            writer.add_scalar(
                "replay/update_count",
                replay_update_count,
                global_step,
            )
            writer.add_scalar(
                "replay/approx_kl",
                replay_last_kl,
                global_step,
            )
            writer.add_scalar(
                "replay/sample_weight_min",
                float(sampled_weights.min().item()),
                global_step,
            )
            writer.add_scalar(
                "replay/sample_weight_max",
                float(sampled_weights.max().item()),
                global_step,
            )
            writer.add_scalar(
                "replay/sample_index",
                int(sampled_indices[0]),
                global_step,
            )
            # Recompute rollout priorities after replay optimization.
            updated_priorities = []

            with torch.no_grad():
                for rollout in sampled_rollouts:
                    rollout_obs = rollout["obs"].to(device)
                    rollout_returns = rollout["returns"].to(device)

                    current_values = agent.get_value(
                        rollout_obs
                    ).view(-1)

                    priority = (
                            current_values - rollout_returns
                    ).abs().mean().item()

                    updated_priorities.append(priority)

            replay_buffer.update_priorities(
                sampled_indices,
                updated_priorities,
            )

            writer.add_scalar(
                "replay/priority_mean",
                float(np.mean(updated_priorities)),
                global_step,
            )
            writer.add_scalar(
                "replay/priority_min",
                float(np.min(updated_priorities)),
                global_step,
            )
            writer.add_scalar(
                "replay/priority_max",
                float(np.max(updated_priorities)),
                global_step,
            )
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
