import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from minesweeper import COVERED


def _groups(n_filters):
    return max(g for g in range(1, min(32, n_filters) + 1) if n_filters % g == 0)


class NoisyLinear(nn.Module):
    """Linear layer with factorised Gaussian noise on its weights (noisy nets).
    The noise is only used in training mode, eval mode uses the mean weights."""
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        bound = in_features ** -0.5
        self.weight_mu = nn.Parameter(th.empty(out_features, in_features).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(th.full((out_features, in_features), sigma0 * bound))
        self.bias_mu = nn.Parameter(th.empty(out_features).uniform_(-bound, bound))
        self.bias_sigma = nn.Parameter(th.full((out_features,), sigma0 * bound))
        self.register_buffer('eps_in', th.zeros(in_features), persistent=False)
        self.register_buffer('eps_out', th.zeros(out_features), persistent=False)
        self.reset_noise()

    @staticmethod
    def _scaled(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_in.copy_(self._scaled(th.randn_like(self.eps_in)))
        self.eps_out.copy_(self._scaled(th.randn_like(self.eps_out)))

    def _weights(self):
        if not self.training:
            return self.weight_mu, self.bias_mu
        weight = self.weight_mu + self.weight_sigma * th.outer(self.eps_out, self.eps_in)
        return weight, self.bias_mu + self.bias_sigma * self.eps_out

    def forward(self, x):
        return F.linear(x, *self._weights())


class NoisyConv1x1(NoisyLinear):
    """NoisyLinear applied to every tile."""
    def forward(self, x):
        weight, bias = self._weights()
        return F.conv2d(x, weight[:, :, None, None], bias)


class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(n_filters), n_filters)
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(n_filters), n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv1(F.relu(self.norm1(x)))
        h = self.conv2(F.relu(self.norm2(h)))
        return x + h


class MinesweeperResCNN(BaseFeaturesExtractor):
    """Fully convolutional residual network with one output per tile. With n_atoms > 1
    each tile outputs a distribution over returns from v_min to v_max (C51); dueling
    adds a board value; noisy makes the output layers noisy."""
    def __init__(self, observation_space, n_filters=128, n_blocks=3, n_atoms=1, v_min=-1.5, v_max=2.5,
                 dueling=False, noisy=False):
        rows, cols = observation_space.shape
        super().__init__(observation_space, features_dim=rows * cols)
        #One-hot input, one channel per tile value
        self.num_classes = int(observation_space.high.max()) + 1
        self.n_atoms = n_atoms
        self.noisy = noisy
        self.dueling = dueling

        layers = [nn.Conv2d(self.num_classes, n_filters, kernel_size=3, padding=1)]
        layers += [ResidualBlock(n_filters) for _ in range(n_blocks)]
        #Pre-activation blocks end on a raw sum, so normalise once before the head
        layers += [nn.GroupNorm(_groups(n_filters), n_filters), nn.ReLU()]
        self._head = len(layers)
        layers += [NoisyConv1x1(n_filters, n_atoms) if noisy
                   else nn.Conv2d(n_filters, n_atoms, kernel_size=1, padding=0)]
        layers += [nn.Flatten()] if n_atoms == 1 else []
        self.conv = nn.Sequential(*layers)
        if n_atoms > 1:
            self.register_buffer('support', th.linspace(v_min, v_max, n_atoms))
        if dueling:
            #Board value from the average features, unchanged by turning the board
            self.value_head = NoisyLinear(n_filters, n_atoms) if noisy else nn.Linear(n_filters, n_atoms)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def _one_hot(self, observation):
        return F.one_hot(observation.to(th.int64), self.num_classes).permute(0, 3, 1, 2).float()

    def _per_tile(self, observation):
        """Shape (batch, tiles, n_atoms): Q-values, or return logits when n_atoms > 1."""
        x = self._one_hot(observation)
        if not self.dueling:
            out = self.conv(x)
            return out.unsqueeze(-1) if self.n_atoms == 1 else out.flatten(2).transpose(1, 2)
        trunk = self.conv[:self._head](x)
        advantage = self.conv[self._head](trunk).flatten(2).transpose(1, 2)
        value = self.value_head(trunk.mean(dim=(2, 3)))
        #Centred on covered tiles only, the moves that exist
        covered = (observation == COVERED).flatten(1).float().unsqueeze(-1)
        mean_advantage = (advantage * covered).sum(dim=1) / covered.sum(dim=1).clamp(min=1)
        return value.unsqueeze(1) + advantage - mean_advantage.unsqueeze(1)

    def forward(self, observation):
        out = self._per_tile(observation)
        return out.squeeze(-1) if self.n_atoms == 1 else self.q_from_logits(out)

    def logits(self, observation):
        return self._per_tile(observation)

    def q_from_logits(self, logits):
        return (logits.softmax(dim=-1) * self.support).sum(dim=-1)


class ConvQNetwork(QNetwork):
    """Uses the extractor's per-tile output as the Q-values, without SB3's MLP head."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        action_dim = int(self.action_space.n)
        if self.features_dim != action_dim:
            raise ValueError(f"ConvQNetwork needs one feature per action, got "
                             f"{self.features_dim} features for {action_dim} actions")
        self.q_net = nn.Identity()


class MinesweeperPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return ConvQNetwork(**net_args).to(self.device)
