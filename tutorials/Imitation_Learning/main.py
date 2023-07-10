from src.enviroments import MiniGameEnv
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter, functional as F
from torch.distributions import Independent, Normal

from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x, dtype=torch.float32, device=DEVICE):
    if isinstance(x, list) and len(x) > 0 and not isinstance(x[0], torch.Tensor):
        x = np.array(x)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    return x.to(dtype).to(device)


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


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


class PolicyNetworkBase(nn.Module):
    # 現状から次の行動を決定する
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def get_distribution(self, states) -> torch.distributions.distribution.Distribution:
        raise NotImplementedError()

    def forward(
        self, states
    ) -> torch.distributions.distribution.Distribution:  # 状態をもとにアクションを決定
        states = to_tensor(states)
        states = torch.sigmoid(states)
        dist = self.get_distribution(states)
        return dist


class DiscreatePolicyNetwork(PolicyNetworkBase):
    # 離散的な行動をとる
    def get_distribution(self, states):
        # print(states)
        pred = self.net(states)
        probs = torch.softmax(pred, dim=-1)
        return torch.distributions.Categorical(probs, validate_args=False)


class AgentPolicy(DiscreatePolicyNetwork):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=256,
        dropout=0,
    ):
        state_size = np.prod(state_size)
        action_size = np.prod(action_size)
        super(AgentPolicy, self).__init__(state_size, action_size, hidden_size)


class ValueNetwork(nn.Module):
    # 状態から価値を予測する
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states):
        states = to_tensor(states, dtype=torch.float32)
        return self.net(states)


class DiscreteDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=4):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_emb = nn.Embedding(action_dim, state_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_logits(self, states, actions):
        states = to_tensor(states).view(-1, self.state_dim)
        actions = to_tensor(actions)
        actions = self.action_emb(actions.long())
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))


class GAIL(nn.Module):
    def __init__(
        self,
        state_size=128,
        action_size=4,
        policy_hidden_dim=64,
        value_hidden_dim=4,
    ):
        super().__init__()

        self.state_dim = np.prod(state_size)
        self.action_dim = np.prod(action_size)

        self.policy_net = AgentPolicy(
            self.state_dim, self.action_dim, hidden_size=policy_hidden_dim
        ).to(DEVICE)
        self.value_net = ValueNetwork(self.state_dim, hidden_dim=value_hidden_dim).to(
            DEVICE
        )
        self.discriminator = DiscreteDiscriminator(
            self.state_dim, self.action_dim, hidden_dim=value_hidden_dim
        ).to(DEVICE)

    def get_action(self, states):
        with torch.inference_mode():
            dist = self.policy_net(states)  # type: ignore
        return dist.sample().detach().cpu().numpy()

    def train(self, env, expert):
        num_steps_per_epochs = 10000
        num_epochs = 100
        gae_gamma = 0.99
        gae_lambda = 0.99
        max_kl = 0.03
        epsilon = 0.02
        normarize_advantage = True
        lambda_ = 0.001
        cg_damping = 0.1

        agent_observations = []
        agent_actions = []
        agent_rewards = []
        max_reward = -np.inf
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters())

        step = 0
        while step < num_steps_per_epochs:
            if step > 1000:
                break

            observation, _ = env.reset()
            agent_rewards_step = []
            agent_observations_step = []
            terminated, truncated = False, False
            reward = -np.inf

            while (
                (not terminated) and (not truncated) and (step < num_steps_per_epochs)
            ):
                env.render(update_interval=10)
                action = expert.get_action()

                agent_observations_step.append(observation)
                agent_observations.append(observation)
                agent_actions.append(action)

                observation, reward, terminated, truncated, info = env.step(action)
                agent_rewards_step.append(reward)
                step += 1
            print(step, reward)

            agent_rewards.append(torch.sum(to_tensor(agent_rewards_step)))
            agent_observations_step = torch.from_numpy(
                np.stack(agent_observations_step)
            )
            agent_rewards_step = to_tensor(agent_rewards_step)

        agent_rewards_mean = to_tensor(agent_rewards).mean()
        max_reward = max(max_reward, agent_rewards_mean.item())
        print(f"agent_rewards_mean: {agent_rewards_mean}")

        agent_observations = torch.from_numpy(np.stack(agent_observations))
        agent_actions = to_tensor(agent_actions, dtype=torch.long)
        reward_epoch_means = []

        for i in range(num_epochs):
            rewards_epochs = []
            observations_epochs = []
            actions_epochs = []
            returns_epochs = []
            advantages_epochs = []
            gae_gammas_epochs = []

            steps = 0
            while steps < num_steps_per_epochs:
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
                    and (steps < num_steps_per_epochs)
                ):
                    env.render(update_interval=10)
                    action = self.get_action(
                        torch.from_numpy(observation.astype(np.float32))
                        .to(DEVICE)
                        .unsqueeze(0)
                    ).item()
                    observation, reward, terminated, truncated, info = env.step(action)

                    observations_step.append(observation)
                    observations_epochs.append(observation)

                    actions_step.append(action)
                    actions_epochs.append(action)

                    rewards_step.append(reward)
                    gae_gammas_step.append(gae_gamma**t)
                    lambdas_step.append(gae_lambda**t)

                    t += 1
                    steps += 1

                # print(f"step: {steps}, observation_text: {observation['text']}, label: {observation['true']}")
                rewards_epochs.append(torch.sum(to_tensor(rewards_step)))
                rewards_step = to_tensor(rewards_step)
                observations_step = torch.from_numpy(np.stack(observations_step))
                actions_step = to_tensor(actions_step, dtype=torch.long)
                gae_gammas_step = to_tensor(gae_gammas_step)
                lambdas_step = to_tensor(lambdas_step)
                costs_step = (
                    -torch.log(
                        self.discriminator(
                            observations_step.reshape(len(actions_step), -1),
                            actions_step,
                        )
                    )
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
                current_values = self.value_net(
                    observations_step.reshape(len(actions_step), -1)
                ).detach()
                next_values = torch.cat(
                    (current_values[1:], to_tensor([[0.0]]))
                ).detach()
                deltas_step = (
                    costs_step.unsqueeze(-1) + gae_gamma * next_values - current_values
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

            observations_epochs = torch.from_numpy(np.stack(observations_epochs))
            actions_epochs = to_tensor(actions_epochs, dtype=torch.long)
            returns_epochs = torch.cat(returns_epochs)
            advantages_epochs = torch.cat(advantages_epochs)
            gae_gammas_epochs = torch.cat(gae_gammas_epochs)

            if normarize_advantage:
                advantages_epochs = (
                    advantages_epochs - advantages_epochs.mean()
                ) / advantages_epochs.std()

            self.discriminator.train()
            novelty_scores = self.discriminator.get_logits(
                observations_epochs, actions_epochs
            )
            agent_scores = self.discriminator.get_logits(
                agent_observations.reshape(len(agent_actions), -1), agent_actions
            )

            self.discriminator_optimizer.zero_grad()
            print(
                reward_epoch_means[-1].item(),
                (reward_epoch_means[-1] * 10 / agent_rewards_mean).item(),
            )
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    agent_scores, torch.zeros_like(agent_scores)
                )
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    novelty_scores, torch.ones_like(novelty_scores)
                )
                - reward_epoch_means[-1] / agent_rewards_mean
            )
            loss.backward(retain_graph=True)
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
            prev_values_epochs = self.value_net(
                observations_epochs.reshape(len(actions_epochs), -1)
            ).detach()
            grad_diffs = get_flat_grads(
                (
                    prev_values_epochs
                    - self.value_net(
                        observations_epochs.reshape(len(actions_epochs), -1)
                    )
                )
                .pow(2)
                .mean(),
                self.value_net,
            )
            g = get_flat_grads(
                (
                    (-1)
                    * (
                        self.value_net(
                            observations_epochs.reshape(len(actions_epochs), -1)
                        ).squeeze()
                        - returns_epochs
                    ).pow(2)
                ).mean(),
                self.value_net,
            )

            def Hv(v):
                return get_flat_grads(torch.dot(grad_diffs, v), self.value_net).detach()

            s = conjugate_gradient(Hv, g)
            Hs = Hv(s)
            alpha = torch.sqrt(2 * epsilon / torch.dot(s, Hs))

            new_params = pred_values_params + alpha * s
            set_params(self.value_net, new_params)

            self.policy_net.train()
            prev_policy_params = get_flat_params(self.policy_net).detach()
            prev_distribution = self.policy_net(
                observations_epochs.reshape(len(actions_epochs), -1)
            )

            def L():
                distb = self.policy_net(
                    observations_epochs.reshape(len(actions_epochs), -1)
                )
                return (
                    advantages_epochs
                    * torch.exp(
                        distb.log_prob(actions_epochs)
                        - prev_distribution.log_prob(actions_epochs).detach()
                    )
                ).mean()

            def kld():
                distb = self.policy_net(
                    observations_epochs.reshape(len(actions_epochs), -1)
                )
                old_probs = prev_distribution.probs.detach()
                probs = distb.probs
                return (
                    (old_probs * (torch.log(old_probs) - torch.log(probs)))
                    .sum(-1)
                    .mean()
                )

            grad_kld_old_param = get_flat_grads(kld(), self.policy_net)

            def Hv(v):
                return (
                    get_flat_grads(
                        torch.dot(grad_kld_old_param, v), self.policy_net
                    ).detach()
                    + cg_damping * v
                )

            g = get_flat_grads(L(), self.policy_net).detach()
            s = conjugate_gradient(Hv, g)
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g,
                s,
                Hs,
                max_kl,
                L,
                kld,
                prev_policy_params,
                self.policy_net,
            )

            disc_causal_entropy = (
                (-1)
                * gae_gammas_epochs
                * self.policy_net(
                    observations_epochs.reshape(len(actions_epochs), -1)
                ).log_prob(actions_epochs)
            ).mean()

            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.policy_net
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.policy_net, new_params)
        return agent_rewards_mean, reward_epoch_means


def get_user_input(env):
    while True:
        event = env.pygame["pygame"].event.wait()
        if event.type == env.pygame["pygame"].QUIT or (
            event.type == env.pygame["pygame"].KEYDOWN
            and event.key == env.pygame["pygame"].K_ESCAPE
        ):
            env.pygame["pygame"].quit()
            exit()

        if event.type == env.pygame["pygame"].KEYDOWN:
            if event.key == env.pygame["pygame"].K_LEFT:
                return 1
            elif event.key == env.pygame["pygame"].K_RIGHT:
                return 0
            elif event.key == env.pygame["pygame"].K_UP:
                return 3
            elif event.key == env.pygame["pygame"].K_DOWN:
                return 2


class Expert:
    def __init__(self, env) -> None:
        self.env = env

    def get_action(self):
        # return self.env.action_space.sample()
        return get_user_input(self.env)


if __name__ == "__main__":
    env = MiniGameEnv(
        limit_frame=10000,
        render_mode="human",
        field_size=(12, 8),
        flag_num=10,
        whole_num=16,
        capture_path="learning.mp4",
    )

    gail = GAIL(env.observation_size)
    expert = Expert(env)
    gail.to(DEVICE)
    gail.train(env, expert)
    # is_play = False

    # if is_play:
    #     expert_data: dict[str, dict] = dict()
    # else:
    #     expert_data = {
    #         k: v.item()
    #         for k, v in np.load("expert_data.npz", allow_pickle=True).items()
    #     }
    # # print(expert_data)
    # class Agent(nn.Module):
    #     def __init__(self, env: MiniGameEnv):
    #         super().__init__()
    #         self.layers = nn.Sequential(
    #             nn.Linear(int(np.prod(env.observation_size)), 256),
    #             nn.Tanh(),
    #             nn.Linear(256, 256),
    #             nn.Tanh(),
    #             nn.Linear(256, env.action_size),
    #         )

    #     def forward(self, x):
    #         x = torch.from_numpy(x.reshape(-1)).float()
    #         return self.layers(x)

    # agent = Agent(env)
    # optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    # c = 0
    # for episode in iter(lambda: c, None):
    #     c += 1
    #     observation, info = env.reset()
    #     while True:
    #         env.render(update_interval=10)
    #         action = agent(observation)
    #         optimizer.zero_grad()
    #         action = torch.softmax(action, dim=0)
    #         observation, reward, terminated, truncated, info = env.step(
    #             torch.argmax(action)
    #             # if action.max() > 0.5
    #             # else print(f"{episode=} | random!") or env.action_space.sample()
    #         )

    #         # 右 左 下 上
    #         x, y = env.current_coord
    #         next_c = [
    #             (min(x + 1, env.field_size[0] - 1), y),
    #             (max(x - 1, 0), y),
    #             (x, min(y + 1, env.field_size[1] - 1)),
    #             (x, max(y - 1, 0)),
    #         ]
    #         next_c = [[1, 2, 0, 1, 4][env.layers[0].field[c]] for c in next_c]
    #         norm_next_c = torch.softmax(
    #             torch.tensor(next_c, dtype=torch.float32, requires_grad=True), dim=0
    #         )
    #         loss = F.mse_loss(action, norm_next_c) + 1 / F.sigmoid(
    #             torch.tensor(reward / 10)
    #         )
    #         loss.backward()
    #         optimizer.step()

    #         if terminated or truncated:
    #             break

    #     if is_play:
    #         expert_data[episode_str] = dict()
    #         expert_data[episode_str]["start_coord"] = env.start_coord
    #         expert_data[episode_str]["goal_coord"] = env.goal_coord
    #         expert_data[episode_str]["init_observation"] = observation
    #         expert_data[episode_str]["actions"] = []

    #     else:
    #         env.from_data(
    #             expert_data[episode_str]["start_coord"],
    #             expert_data[episode_str]["goal_coord"],
    #             expert_data[episode_str]["init_observation"],
    #         )

    #     action = 0
    #     step = 0
    #     reward, terminated, truncated, info = -np.inf, False, False, None
    #     while True:
    #         if is_play:
    #             env.render(update_interval=0)
    #             event = env.pygame["pygame"].event.wait()
    #             if event.type == env.pygame["pygame"].KEYDOWN:
    #                 if event.key == env.pygame["pygame"].K_LEFT:
    #                     action = 1
    #                 elif event.key == env.pygame["pygame"].K_RIGHT:
    #                     action = 0
    #                 elif event.key == env.pygame["pygame"].K_UP:
    #                     action = 3
    #                 elif event.key == env.pygame["pygame"].K_DOWN:
    #                     action = 2
    #             else:
    #                 continue
    #             observation, reward, terminated, truncated, info = env.step(action)

    #         elif is_training:
    #             env.render(update_interval=100)
    #             action_prob = agent(
    #                 torch.from_numpy(observation.flatten().astype(np.float32))
    #             )
    #             action_prob = F.softmax(action_prob, dim=-1)
    #             action = action_prob.argmax().item()
    #             loss = reward
    #         else:
    #             env.render(update_interval=100)
    #             action = expert_data[episode_str]["actions"][step]
    #             step += 1
    #             observation, reward, terminated, truncated, info = env.step(action)

    #         if is_play:
    #             expert_data[episode_str]["actions"].append(action)

    #         if terminated or truncated:
    #             break

    #     if is_play:
    #         expert_data[episode_str]["actions"] = np.array(
    #             expert_data[episode_str]["actions"]
    #         )
    # if is_play:
    #     np.savez("expert_data.npz", **expert_data)  # type: ignore
