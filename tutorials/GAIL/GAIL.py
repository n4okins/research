# %%
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from pathlib import Path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    from torch.cuda import FloatTensor

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


# %%
def to_tensor(x, dtype=torch.float32, device=DEVICE):
    if isinstance(x, list) and len(x) > 0 and not isinstance(x[0], torch.Tensor):
        x = np.array(x)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    return x.to(device)


def get_flat_params(module: nn.Module):
    return torch.cat([param.view(-1) for param in module.parameters()])


def set_params(module: nn.Module, new_flat_params):
    start_idx = 0
    for param in module.parameters():
        end_idx = int(start_idx + np.prod(list(param.shape)))
        param.data = torch.reshape(new_flat_params[start_idx:end_idx], param.shape)
        start_idx = end_idx


def get_flat_grads(f, net):
    flat_grads = torch.cat(
        [
            grad.view(-1)
            for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
        ]
    )
    return flat_grads


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
    g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10, success_ratio=0.1
):
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params


# %%
class PolicyNetworkBase(nn.Module):
    # 現状から次の行動を決定する
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def get_distribution(self, states) -> torch.distributions.distribution.Distribution:
        raise NotImplementedError()

    def forward(
        self, states
    ) -> torch.distributions.distribution.Distribution:  # 状態をもとにアクションを決定
        states = to_tensor(states)
        return self.get_distribution(states)


class DiscreatePolicyNetwork(PolicyNetworkBase):
    # 離散的な行動をとる
    def get_distribution(self, states):
        probs = torch.softmax(self.net(states), dim=-1)
        return torch.distributions.Categorical(probs)


class ContinuousPolicyNetwork(PolicyNetworkBase):
    # 連続的な行動をとる
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_distribution(self, states):
        mean = self.net(states)
        std = torch.exp(self.log_std)
        cov_mtx = torch.eye(self.action_dim) * (std**2)
        return torch.distributions.MultivariateNormal(mean, cov_mtx)


# %%
class ValueNetwork(nn.Module):
    # 状態から価値を予測する
    def __init__(self, state_dim, hidden_dim=50):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states):
        states = to_tensor(states)
        return self.net(states)


# %%
class DiscreteDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super().__init__()
        self.action_emb = nn.Embedding(action_dim, state_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_logits(self, states, actions):
        states = to_tensor(states)
        actions = to_tensor(actions)
        actions = self.action_emb(actions.long())
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))


class ContinuousDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_logits(self, states, actions):
        states = to_tensor(states)
        actions = to_tensor(actions)
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))


# %%
class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, is_discrete, policy_hidden_dim=50):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete

        self.policy_net = (
            DiscreatePolicyNetwork(state_dim, action_dim, hidden_dim=policy_hidden_dim)
            if is_discrete
            else ContinuousPolicyNetwork(
                state_dim, action_dim, hidden_dim=policy_hidden_dim
            )
        )

    def get_action(self, states):
        self.policy_net.eval()
        dist = self.policy_net(states)
        return dist.sample().detach().cpu().numpy()


# %%
@dataclass
class GAILParams:
    num_epochs = 200
    num_steps_per_epochs = 1000
    lambda_ = 1e-3
    gae_gamma = 0.99
    gae_lambda = 0.99
    epsilon = 0.01
    max_kl = 0.01
    cg_damping = 0.1
    normarize_advantage = True

    policy_net_hidden_dim = 50
    value_net_hidden_dim = 50
    discriminator_hidden_dim = 50


@dataclass
class GAILConfig(DictConfig):
    name: str
    params: GAILParams = GAILParams()


def config_to_yaml(cfg: DictConfig):
    return OmegaConf.to_yaml(cfg)


# %%


class GAIL(nn.Module):
    def __init__(self, state_dim, action_dim, is_discrete: bool, params: GAILParams):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.params = params

        self.policy_net = (
            DiscreatePolicyNetwork(
                state_dim, action_dim, hidden_dim=params.policy_net_hidden_dim
            )
            if is_discrete
            else ContinuousPolicyNetwork(
                state_dim, action_dim, hidden_dim=params.policy_net_hidden_dim
            )
        )
        self.value_net = ValueNetwork(state_dim, params.value_net_hidden_dim)
        self.discriminator = (
            DiscreteDiscriminator(
                state_dim, action_dim, params.discriminator_hidden_dim
            )
            if is_discrete
            else ContinuousDiscriminator(
                state_dim, action_dim, params.discriminator_hidden_dim
            )
        )

    def get_action(self, states):
        with torch.inference_mode():
            dist = self.policy_net(states)
        return dist.sample().detach().cpu().numpy()

    def train(self, env: gym.Env, agent: Agent, render=False):
        agent_observations = []
        agent_actions = []
        agent_rewards = []
        max_reward = -np.inf
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters())

        step = 0
        while step < self.params.num_steps_per_epochs:
            observation, _ = env.reset()
            agent_rewards_step = []
            agent_observations_step = []
            terminated, truncated = False, False

            while (
                (not terminated)
                and (not truncated)
                and (step < self.params.num_steps_per_epochs)
            ):
                action = agent.get_action(observation)

                agent_observations_step.append(observation)
                agent_observations.append(observation)
                agent_actions.append(action)

                observation, reward, terminated, truncated, info = env.step(action)
                agent_rewards_step.append(reward)
                if render:
                    env.render()

                step += 1

            agent_rewards.append(torch.sum(to_tensor(agent_rewards_step)))
            agent_observations_step = to_tensor(agent_observations_step)
            agent_rewards_step = to_tensor(agent_rewards_step)

        agent_rewards_mean = to_tensor(agent_rewards).mean()
        max_reward = max(max_reward, agent_rewards_mean.item())
        print(f"agent_rewards_mean: {agent_rewards_mean}")
        agent_observations = to_tensor(agent_observations)
        agent_actions = to_tensor(agent_actions, dtype=torch.long)
        reward_epoch_means = []

        for i in range(self.params.num_epochs):
            rewards_epochs = []
            observations_epochs = []
            actions_epochs = []
            returns_epochs = []
            advantages_epochs = []
            gae_gammas_epochs = []

            steps = 0
            while steps < self.params.num_steps_per_epochs:
                rewards_step = []
                observations_step = []
                actions_step = []
                gae_gammas_step = []
                lambdas_step = []
                t = 0
                terminated, truncated = False, False

                observation, _ = env.reset()

                while (
                    (not terminated)
                    and (not truncated)
                    and (steps < self.params.num_steps_per_epochs)
                ):
                    action = self.get_action(observation)
                    observation, reward, terminated, truncated, info = env.step(action)

                    observations_step.append(observation)
                    observations_epochs.append(observation)

                    actions_step.append(action)
                    actions_epochs.append(action)

                    if render:
                        env.render()

                    rewards_step.append(reward)
                    gae_gammas_step.append(self.params.gae_gamma**t)
                    lambdas_step.append(self.params.gae_lambda**t)

                    t += 1
                    steps += 1

                rewards_epochs.append(torch.sum(to_tensor(rewards_step)))
                rewards_step = to_tensor(rewards_step)
                observations_step = to_tensor(observations_step)
                actions_step = to_tensor(actions_step, dtype=torch.long)
                gae_gammas_step = to_tensor(gae_gammas_step)
                lambdas_step = to_tensor(lambdas_step)
                costs_step = (
                    -torch.log(self.discriminator(observations_step, actions_step))
                    .squeeze()
                    .detach()
                )
                discriminator_costs_step = gae_gammas_step * costs_step
                discriminator_returns_step = to_tensor(
                    [discriminator_costs_step[i:].sum() for i in range(t)]
                )
                returns_step = discriminator_returns_step / gae_gammas_step
                returns_epochs.append(returns_step)

                self.value_net.eval()
                current_values = self.value_net(observations_step).detach()
                next_values = torch.cat(
                    (self.value_net(observations_step)[1:], to_tensor([[0.0]]))
                ).detach()
                deltas_step = (
                    costs_step.unsqueeze(-1)
                    + self.params.gae_gamma * next_values
                    - current_values
                )
                advantages_step = to_tensor(
                    [
                        (
                            (gae_gammas_step * lambdas_step)[: t - j].unsqueeze(-1)
                            * deltas_step[j:]
                        ).sum()
                        for j in range(t)
                    ]
                )
                advantages_epochs.append(advantages_step)
                gae_gammas_epochs.append(gae_gammas_step)

            reward_epoch_means.append(to_tensor(rewards_epochs).mean())

            observations_epochs = to_tensor(observations_epochs)
            actions_epochs = to_tensor(actions_epochs, dtype=torch.long)
            returns_epochs = torch.cat(returns_epochs)
            advantages_epochs = torch.cat(advantages_epochs)
            gae_gammas_epochs = torch.cat(gae_gammas_epochs)

            if self.params.normarize_advantage:
                advantages_epochs = (
                    advantages_epochs - advantages_epochs.mean()
                ) / advantages_epochs.std()

            self.discriminator.train()
            novelty_scores = self.discriminator.get_logits(
                observations_epochs, actions_epochs
            )
            agent_scores = self.discriminator.get_logits(
                agent_observations, agent_actions
            )

            self.discriminator_optimizer.zero_grad()
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    agent_scores, torch.zeros_like(agent_scores)
                )
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    novelty_scores, torch.ones_like(novelty_scores)
                )
                + agent_rewards_mean / (reward_epoch_means[-1] + 1)
            )
            loss.backward()
            self.discriminator_optimizer.step()

            print(
                f"Epoch {i + 1},  Reward Mean: {reward_epoch_means[-1]}, Loss: {loss.item():.4f}"
            )

            if reward_epoch_means[-1] > max_reward:
                print(f"- Saving model with reward {reward_epoch_means[-1]}")
                torch.save(
                    self.policy_net.state_dict(),
                    Path(__file__).parent / "policy_net.pt",
                )
                max_reward = reward_epoch_means[-1]

            self.value_net.train()
            pred_values_params = get_flat_params(self.value_net).detach()
            prev_values_epochs = self.value_net(observations_epochs).detach()
            grad_diffs = get_flat_grads(
                (prev_values_epochs - self.value_net(observations_epochs))
                .pow(2)
                .mean(),
                self.value_net,
            )
            g = get_flat_grads(
                (
                    (-1)
                    * (
                        self.value_net(observations_epochs).squeeze() - returns_epochs
                    ).pow(2)
                ).mean(),
                self.value_net,
            )

            def Hv(v):
                return get_flat_grads(torch.dot(grad_diffs, v), self.value_net).detach()

            s = conjugate_gradient(Hv, g)
            Hs = Hv(s)
            alpha = torch.sqrt(2 * self.params.epsilon / torch.dot(s, Hs))

            new_params = pred_values_params + alpha * s
            set_params(self.value_net, new_params)

            self.policy_net.train()
            prev_policy_params = get_flat_params(self.policy_net).detach()
            prev_distribution = self.policy_net(observations_epochs)

            def L():
                distb = self.policy_net(observations_epochs)
                return (
                    advantages_epochs
                    * torch.exp(
                        distb.log_prob(actions_epochs)
                        - prev_distribution.log_prob(actions_epochs).detach()
                    )
                ).mean()

            def kld():
                distb = self.policy_net(observations_epochs)
                if self.is_discrete:
                    old_probs = prev_distribution.probs.detach()
                    probs = distb.probs
                    return (
                        (old_probs * (torch.log(old_probs) - torch.log(probs)))
                        .sum(-1)
                        .mean()
                    )
                else:
                    old_mean = prev_distribution.mean.detach()
                    old_cov = prev_distribution.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                        (old_cov / cov).log().sum(-1)
                        + ((old_cov + (old_mean - mean).pow(2)) / cov).sum(-1)
                        - self.action_dim
                        + torch.log(cov).sum(-1)
                        - torch.log(old_cov).sum(-1)
                    ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.policy_net)

            def Hv(v):
                return (
                    get_flat_grads(
                        torch.dot(grad_kld_old_param, v), self.policy_net
                    ).detach()
                    + self.params.cg_damping * v
                )

            g = get_flat_grads(L(), self.policy_net).detach()
            s = conjugate_gradient(Hv, g)
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g,
                s,
                Hs,
                self.params.max_kl,
                L,
                kld,
                prev_policy_params,
                self.policy_net,
            )

            disc_causal_entropy = (
                (-1)
                * gae_gammas_epochs
                * self.policy_net(observations_epochs).log_prob(actions_epochs)
            ).mean()

            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.policy_net
            )
            new_params += self.params.lambda_ * grad_disc_causal_entropy

            set_params(self.policy_net, new_params)
        return agent_rewards_mean, reward_epoch_means

    def save_state(self):
        state = dict()
        if hasattr(self, "policy_net"):
            state["policy_net"] = self.policy_net.state_dict()
        if hasattr(self, "value_net"):
            state["value_net"] = self.value_net.state_dict()
        if hasattr(self, "discriminator"):
            state["discriminator"] = self.discriminator.state_dict()
        if hasattr(self, "discriminator_optimizer"):
            state["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()

        state["params"] = self.params
        torch.save(state, Path(__file__).parent / "state.pth")

    def load_state(self, state_path: Path):
        if state_path.exists():
            state = torch.load(state_path)
            if hasattr(self, "policy_net"):
                self.policy_net.load_state_dict(state["policy_net"])
            if hasattr(self, "value_net"):
                self.value_net.load_state_dict(state["value_net"])
            if hasattr(self, "discriminator"):
                self.discriminator.load_state_dict(state["discriminator"])
            if hasattr(self, "discriminator_optimizer"):
                self.discriminator_optimizer.load_state_dict(
                    state["discriminator_optimizer"]
                )
            self.params = state["params"]


# %%
@hydra.main(version_base=None, config_name="GAIL_config", config_path=str(Path(__file__).parent))
def run(cfg: GAILConfig):
    import pybullet_envs

    print(config_to_yaml(cfg))

    env = gym.make(cfg.name, render_mode="human")
    env.reset()
    print(
        f"State: {env.observation_space}, {env.observation_space.shape}\n Action: {env.action_space}, {env.action_space.shape}"
    )
    state_dim = env.observation_space.shape[0]
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if is_discrete:
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, action dim: {action_dim}")

    agent = Agent(
        state_dim,
        action_dim,
        is_discrete=is_discrete,
        policy_hidden_dim=cfg.params.policy_net_hidden_dim,
    ).to(DEVICE)

    # agent.policy_net.load_state_dict(
    #     # torch.load(Path(__file__).parent / "policy_net.pt")
    #     torch.load(
    #         "/home/n4okins/repositories/research_2/clone_repos/gail-pytorch/experts/Pendulum-v0/policy.ckpt"
    #     )
    # )

    model = GAIL(state_dim, action_dim, is_discrete=is_discrete, params=cfg.params).to(
        DEVICE
    )
    results = model.train(env, agent, render=False)

    # for i in range(100):
    #     obs, _ = env.reset()
    #     reward = 0
    #     terminated, truncated = False, False
    #     while not terminated and not truncated:
    #         # env.render()
    #         action = agent.get_action(obs)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #     print(f"Episode {i}, reward: {reward}")

    env.close()


if __name__ == "__main__":
    run()
