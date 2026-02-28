import collections
import logging

import open_spiel
from open_spiel.python.algorithms import exploitability as _
from open_spiel.python.pytorch import deep_cfr as _
import numpy as np
import torch
import torch.utils.tensorboard as _

import util


from torch import nn
MLP = open_spiel.python.pytorch.deep_cfr.MLP
SpielReservoirBuffer = open_spiel.python.pytorch.deep_cfr.ReservoirBuffer


class Config:
    def __init__(self):
        self.embedding_size = 64
        self.memory_capacity = 1e6

        self.value_traversals = -1
        self.value_batch_size = -1
        self.value_batch_steps = -1
        self.value_learning_rate = 1e-3

        self.regret_traversals = 1024
        self.regret_batch_size = 256
        self.regret_batch_steps = 375
        self.regret_learning_rate = 1e-3

        self.avg_policy_batch_size = 256
        self.avg_policy_batch_steps = 2500
        self.avg_policy_learning_rate = 1e-3


class Agent:
    def __init__(self, game, cfg):
        self.cfg = cfg
        self.t = 1
        obs_dim = game.new_initial_state().information_state_tensor().shape[0]
        action_dim = game.num_distinct_actions()

        self._num_actions = action_dim

        self.avg_policy_buffer = ReservoirBuffer(cfg.memory_capacity)
        self.avg_policy_net = MLP(obs_dim, [64,], action_dim)

        self.regret_buffers = [ReservoirBuffer(cfg.memory_capacity) for _ in range(game.num_players())]
        self.regret_nets = [MLP(obs_dim, [64,], action_dim) for _ in range(game.num_players())]

        self.avg_policy_t = 0
        self.regret_t = 0
        self.value_t = 0

    def action_probabilities(self, obs, mask_np):
        # device = self.avg_policy_net.linear.weight.device
        device = torch.device("cpu")
        with torch.no_grad():
            x = torch.from_numpy(obs).to(torch.float32).to(device)
            mask = torch.from_numpy(mask_np).to(device)

            logits = self.avg_policy_net(x)
            probs = torch.nn.functional.softmax(logits, dim=0)

            probs = torch.mul(probs, mask)
            probs = probs / torch.sum(probs)

        return probs.cpu().numpy()


def _train_avg_policy(cfg, agent):
    num_epoch = 8
    epoch_steps = int(np.ceil(agent.cfg.avg_policy_batch_steps / num_epoch))
    dataloader_train = torch.utils.data.DataLoader(
            agent.avg_policy_buffer, batch_size=agent.cfg.avg_policy_batch_size, shuffle=True)
    optimizer = torch.optim.Adam(agent.avg_policy_net.parameters(), lr=agent.cfg.avg_policy_learning_rate)

    for _ in range(num_epoch):
        agent.avg_policy_t += 1
        metrics = {}

        agent.avg_policy_net.train()
        for _ in range(epoch_steps):
            batch = next(iter(dataloader_train))

            loss = _get_avg_policy_loss(agent, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = util.update_metric(metrics, "avg_policy/train/loss", loss)

        for k, v in metrics.items():
            cfg.summary_writer.add_scalar(k, v.compute(), agent.avg_policy_t)


def _gather_regret_data(game, agent, player):
    for _ in range(agent.cfg.regret_traversals):
        state = game.new_initial_state()
        while not state.is_terminal():
            # Get policy.
            current_player = state.current_player()
            obs = state.information_state_tensor()
            mask = state.legal_actions_mask()
            policy = _match_regret(agent, current_player, obs, mask)

            # Add data to buffer.
            if current_player == player:
                # regret = _get_regret(agent, obs, policy)
                regret = _get_regret_exact(agent, state)
                sr = StateRegret(state=obs, regret=regret, mask=mask, t=agent.t)
                agent.regret_buffers[player].add(sr)
            else:
                behaviour = Behaviour(state=obs, policy=policy, t=agent.t)
                agent.avg_policy_buffer.add(behaviour)

            # Update state with policy.
            if current_player == player:
                sample_policy = mask / np.sum(mask)
            else:
                sample_policy = policy
            action = np.random.choice(range(len(sample_policy)), p=sample_policy)
            state = state.child(action)


def _train_regret(cfg, agent):
    for player in range(cfg.game.num_players()):
        _gather_regret_data(cfg.game, agent, player)

        num_epoch = 8
        epoch_steps = int(np.ceil(agent.cfg.regret_batch_steps / num_epoch))
        dataloader_train = torch.utils.data.DataLoader(agent.regret_buffers[player], batch_size=agent.cfg.regret_batch_size, shuffle=True)
        regret_net = agent.regret_nets[player]
        regret_net.reset()
        optimizer = torch.optim.Adam(regret_net.parameters(), lr=agent.cfg.regret_learning_rate)

        for _ in range(num_epoch):
            agent.regret_t += 1
            metrics = {}

            for _ in range(epoch_steps):
                batch = next(iter(dataloader_train))

                loss = _get_regret_loss(agent, player, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics = util.update_metric(metrics, "regret/train/loss", loss)

            for k, v in metrics.items():
                cfg.summary_writer.add_scalar(k, v.compute(), agent.regret_t)


def _get_avg_policy_loss(agent, batch):
    # device = agent.avg_policy_net.linear.weight.device
    device = torch.device("cpu")
    x = batch.state.to(torch.float32).to(device)
    y_policy = batch.policy.to(device)

    logits = agent.avg_policy_net(x)

    loss = torch.nn.functional.cross_entropy(logits, y_policy)

    # Linear CFR.
    weight = batch.t.to(device) / agent.t
    loss = torch.mul(loss, weight)

    return torch.mean(loss)


def _get_regret_loss(agent, player, batch):
    # device = agent.avg_policy_net.linear.weight.device
    device = torch.device("cpu")

    x = batch.state.to(torch.float32).to(device)
    mask = batch.mask.to(device)
    y_regret = batch.regret.to(device)

    regret = agent.regret_nets[player](x)

    loss = torch.pow(regret - y_regret, 2)

    # Linear CFR.
    weight = batch.t.to(device) / agent.t
    weight = weight.unsqueeze(-1).expand(-1, loss.shape[-1])
    loss = torch.mul(loss, weight)

    loss = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
    return loss


def _match_regret(agent, player, obs, mask_np):
    # device = agent.avg_policy_net.linear.weight.device
    device = torch.device("cpu")
    with torch.no_grad():
        x = torch.from_numpy(obs).to(torch.float32).to(device)
        regrets = agent.regret_nets[player](x)
        raw_regrets = regrets.cpu().numpy()

    regrets = np.clip(raw_regrets, a_min=0, a_max=None)
    regrets = regrets * mask_np
    summed = np.sum(regrets)
    if summed > 1e-6:
        return regrets / summed

    # Just use the best regret.
    max_id, max_regret = 0, raw_regrets[0]
    for i, m in enumerate(mask_np):
        if m == 1 and raw_regrets[i] > max_regret:
            max_id, max_regret = i, raw_regrets[i]
    policy = np.zeros(regrets.shape, dtype=regrets.dtype)
    policy[max_id] = 1
    return policy


def _get_regret_exact(agent, state):
    player = state.current_player()

    # Get value of children.
    mask = state.legal_actions_mask()
    children_values = np.zeros(mask.shape, dtype=float)
    for a, m in enumerate(mask):
        if m == 1:
            child = state.child(a)
            children_values[a] = _value_exact(player, agent, child)

    # Get policy.
    obs = state.information_state_tensor()
    policy = _match_regret(agent, player, obs, mask)

    value = np.sum(policy * children_values)
    regret = children_values - value
    return regret


def _value_exact(player, agent, state):
    if state.is_terminal():
        return state.returns()[player]

    # Get children values recursively.
    mask = state.legal_actions_mask()
    children_values = np.zeros(mask.shape, dtype=float)
    for a, m in enumerate(mask):
        if m == 1:
            child = state.child(a)
            children_values[a] = _value_exact(player, agent, child)

    # Get policy.
    current_player = state.current_player()
    obs = state.information_state_tensor()
    policy = _match_regret(agent, current_player, obs, mask)

    value = np.sum(policy * children_values)
    return value


def _calc_nashconv(game, agent):
    def action_probabilities(spiel_state):
        state = game.from_spiel(spiel_state)
        obs = state.information_state_tensor()
        mask = state.legal_actions_mask()
        probs = agent.action_probabilities(obs, mask)
        return {a: probs[a] for a in spiel_state.legal_actions()}

    policy = open_spiel.python.policy.tabular_policy_from_callable(game._game, action_probabilities)
    conv = open_spiel.python.algorithms.exploitability.nash_conv(game._game, policy)
    logging.info("iteration %d nashconv %f", agent.t, conv)
    return conv


class TrainConfig:
    def __init__(self):
        self.run_dir = ""
        self.device_name = ""
        self.game = None
        self.test_agent = None

        # Derived fields.
        self.summary_writer = None
        self.device = None

    def setup(self):
        self.summary_writer = torch.utils.tensorboard.SummaryWriter(self.run_dir)
        self.device = torch.device(self.device_name)


def train(cfg, agent):
    for _ in range(999999):
        _train_regret(cfg, agent)
        agent.t += 1

        if agent.t % 20 == 0:
            _train_avg_policy(cfg, agent)
            _calc_nashconv(cfg.game, agent)


Transition = collections.namedtuple("Transition", ["player", "observation", "importance", "action", "reward"])
StateActionValue = collections.namedtuple(
        "StateActionValue", ["state", "action", "value"])
StateRegret = collections.namedtuple(
        "StateRegret", ["state", "regret", "mask", "t"])
Behaviour = collections.namedtuple(
        "Behaviour", ["state", "policy", "t"])


class ReservoirBuffer(object):
    def __init__(self, capacity):
        self._buf = open_spiel.python.pytorch.deep_cfr.ReservoirBuffer(capacity)

    def add(self, element):
        self._buf.add(element)

    def clear(self):
        self._buf.clear()

    def __len__(self):
        return len(self._buf)

    def __getitem__(self, idx):
        return self._buf._data[idx]
