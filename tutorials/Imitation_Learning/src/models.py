import torch
import torch.nn as nn
from torch.nn import Parameter, functional as F
from torch.distributions import Independent, Normal

ACTIVATION_FUNCTIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


# Concatenates the state and action (previously one-hot discrete version)
def _join_state_action(state, action, action_size):
    return torch.cat([state, action], dim=1)


# Computes the squared distance between two sets of vectors
def _squared_distance(x, y):
    n_1, n_2, d = x.size(0), y.size(0), x.size(1)
    tiled_x, tiled_y = x.view(n_1, 1, d).expand(n_1, n_2, d), y.view(1, n_2, d).expand(
        n_1, n_2, d
    )
    return (tiled_x - tiled_y).pow(2).mean(dim=2)


# Gaussian/radial basis function/exponentiated quadratic kernel
def _gaussian_kernel(x, y, gamma=1):
    return torch.exp(-gamma * _squared_distance(x, y))


# Creates a sequential fully-connected network
def _create_fcnn(
    input_size, hidden_size, output_size, activation_function, dropout=0, final_gain=1.0
):
    assert activation_function in ACTIVATION_FUNCTIONS.keys()

    network_dims, layers = (input_size, hidden_size, hidden_size), []

    for k in range(len(network_dims) - 1):
        layer = nn.Linear(network_dims[k], network_dims[k + 1])
        nn.init.orthogonal_(
            layer.weight, gain=nn.init.calculate_gain(activation_function)  # type: ignore
        )
        nn.init.constant_(layer.bias, 0)
        layers.append(layer)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(ACTIVATION_FUNCTIONS[activation_function]())

    final_layer = nn.Linear(network_dims[-1], output_size)
    nn.init.orthogonal_(final_layer.weight, gain=final_gain)  # type: ignore
    nn.init.constant_(final_layer.bias, 0)
    layers.append(final_layer)

    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size,
        activation_function="tanh",
        log_std_dev_init=-0.5,
        dropout=0,
    ):
        super().__init__()
        self.actor = _create_fcnn(
            state_size,
            hidden_size,
            output_size=action_size,
            activation_function=activation_function,
            dropout=dropout,
            final_gain=0.01,
        )
        self.log_std_dev = Parameter(
            torch.full((action_size,), log_std_dev_init, dtype=torch.float32)
        )

    def forward(self, state):
        mean = self.actor(state)
        policy = Independent(Normal(mean, self.log_std_dev.exp()), 1)
        return policy

    # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        return self.forward(state).log_prob(action)

    def _get_action_uncertainty(self, state, action):
        ensemble_policies = []
        for _ in range(5):  # Perform Monte-Carlo dropout for an implicit ensemble
            ensemble_policies.append(self.log_prob(state, action).exp())
        return torch.stack(ensemble_policies).var(dim=0)

    # Set uncertainty threshold at the 98th quantile of uncertainty costs calculated over the expert data
    def set_uncertainty_threshold(self, expert_state, expert_action):
        self.q = torch.quantile(
            self._get_action_uncertainty(expert_state, expert_action), 0.98
        ).item()

    def predict_reward(self, state, action):
        # Calculate (raw) uncertainty cost
        uncertainty_cost = self._get_action_uncertainty(state, action)
        # Calculate clipped uncertainty cost
        neg_idxs = uncertainty_cost.less_equal(self.q)
        uncertainty_cost[neg_idxs] = -1
        uncertainty_cost[~neg_idxs] = 1
        return -uncertainty_cost


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size, activation_function="tanh"):
        super().__init__()
        self.critic = _create_fcnn(
            state_size,
            hidden_size,
            output_size=1,
            activation_function=activation_function,
        )

    def forward(self, state):
        value = self.critic(state).squeeze(dim=1)
        return value


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size,
        activation_function="tanh",
        log_std_dev_init=-0.5,
        dropout=0,
    ):
        super().__init__()
        self.actor = Actor(
            state_size,
            action_size,
            hidden_size,
            activation_function=activation_function,
            log_std_dev_init=log_std_dev_init,
            dropout=dropout,
        )
        self.critic = Critic(
            state_size, hidden_size, activation_function=activation_function
        )

    def forward(self, state):
        policy, value = self.actor(state), self.critic(state)
        return policy, value

    def get_greedy_action(self, state):
        return self.actor(state).mean

    # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        return self.actor.log_prob(state, action)


class GAILDiscriminator(nn.Module):
    def __init__(
        self, state_size, action_size, hidden_size, state_only=False, forward_kl=False
    ):
        super().__init__()
        self.action_size, self.state_only, self.forward_kl = (
            action_size,
            state_only,
            forward_kl,
        )
        self.discriminator = _create_fcnn(
            state_size if state_only else state_size + action_size,
            hidden_size,
            1,
            "tanh",
        )

    def forward(self, state, action):
        D = self.discriminator(
            state
            if self.state_only
            else _join_state_action(state, action, self.action_size)
        ).squeeze(dim=1)
        return D

    def predict_reward(self, state, action):
        D = torch.sigmoid(self.forward(state, action))
        h = torch.log(D + 1e-6) - torch.log1p(
            -D + 1e-6
        )  # Add epsilon to improve numerical stability given limited floating point precision
        return torch.exp(h) * -h if self.forward_kl else h


if __name__ == "__main__":
    print(torch.cuda.is_available())
    m = nn.Linear(1, 1).cuda()
    print(m(torch.ones(1, 1).cuda()))
