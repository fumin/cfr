import collections
import logging
import math
import os

import open_spiel
from open_spiel.python.algorithms import exploitability as _
from open_spiel.python.pytorch import deep_cfr as _
import numpy as np
import torch
import torch.utils.tensorboard as _

import util


Config = collections.namedtuple("Config", [
    "embedding_size",
    "memory_capacity",

    "value_traversals",
    "value_batch_size",
    "value_batch_steps",
    "value_learning_rate",

    "regret_traversals",
    "regret_batch_size",
    "regret_batch_steps",
    "regret_learning_rate",

    "avg_policy_batch_size",
    "avg_policy_batch_steps",
    "avg_policy_learning_rate",
    ])


def new_config():
    cfg = Config(
            embedding_size=64,
            memory_capacity=1e6,

            # Paper 1000, Code 100
            value_traversals=1000,
            # Paper, batch_size 2048, batch_steps 5000
            # Code, batch_size 256, batch_steps 200
            value_batch_size=256,
            value_batch_steps=256,
            value_learning_rate=1e-3,

            # Paper 1000, code 500
            regret_traversals=500,
            # Paper, batch_size 2048, batch_steps 5000
            # Code, batch_size 256, batch_steps 200
            regret_batch_size=256,
            regret_batch_steps=256,
            regret_learning_rate=1e-3,

            # Paper, batch_size ?, batch_steps 10000
            # Code, batch_size 10000, batch_steps 1000
            avg_policy_batch_size=8192,
            avg_policy_batch_steps=1024,
            avg_policy_learning_rate=1e-3,
            )
    return cfg


class Agent:
    def __init__(self, game, cfg):
        self.cfg = cfg
        self.t = 0
        obs_dim = game.new_initial_state().information_state_tensor().shape[0]
        action_dim = game.num_distinct_actions()
        self.trunk = Trunk(obs_dim, cfg.embedding_size)

        self.avg_policy_net = PolicyNet(self.trunk, action_dim)
        # self.avg_policy_net = PolicyNet(Trunk(obs_dim, cfg.embedding_size), action_dim)
        self.avg_policy_buffer = ReservoirBuffer(cfg.memory_capacity)

        regret_net = PolicyNet(self.trunk, action_dim)
        self.regret_nets = []
        self.regret_buffers = []
        for p in range(game.num_players()):
            self.regret_nets.append(regret_net)
            # self.regret_nets.append(PolicyNet(Trunk(obs_dim, cfg.embedding_size), action_dim))
            self.regret_buffers.append(ReservoirBuffer(cfg.memory_capacity))

        self.value_net = PolicyNet(self.trunk, action_dim)
        # self.value_net = PolicyNet(Trunk(obs_dim, cfg.embedding_size), action_dim)
        self.value_buffer = ReservoirBuffer(cfg.memory_capacity)

        self.avg_policy_t = 0
        self.regret_t = 0
        self.value_t = 0

    def action_probabilities(self, obs, mask_np):
        device = self.avg_policy_net.linear.weight.device
        with torch.no_grad():
            x = torch.from_numpy(obs).to(torch.float32).to(device)
            mask = torch.from_numpy(mask_np).to(device)

            logits = self.avg_policy_net(x)
            probs = torch.nn.functional.softmax(logits, dim=0)

            probs = torch.mul(probs, mask)

        return probs.cpu().numpy()

    def get_action(self, state):
        obs = state.information_state_tensor()
        mask = state.legal_actions_mask()
        probs = self.action_probabilities(obs, mask)
        action = np.argmax(np.random.default_rng().multinomial(1, probs)).item()
        return action


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


def load_checkpoint(agent, checkpoint_dir):
    torch.serialization.add_safe_globals([Behaviour, StateRegret])

    checkpoint, cp_path = util.load_checkpoint(checkpoint_dir)
    if not checkpoint:
        logging.info("no checkpoint")
        return

    agent.t = checkpoint["t"]
    agent.avg_policy_net.load_state_dict(checkpoint["avg_policy_net"])
    for p, _ in enumerate(agent.regret_nets):
        agent.regret_nets[p].load_state_dict(checkpoint["regret_net_{}".format(p)])
    agent.value_net.load_state_dict(checkpoint["value_net"])

    agent.avg_policy_t = checkpoint["avg_policy_t"]
    agent.regret_t = checkpoint["regret_t"]
    agent.value_t = checkpoint["value_t"]

    logging.info("loaded checkpoint %s", cp_path)


def _save_checkpoint(checkpoint_dir, agent):
    cp = {}
    cp["t"] = agent.t
    cp["avg_policy_net"] = agent.avg_policy_net.state_dict()
    for p, _ in enumerate(agent.regret_nets):
        cp["regret_net_{}".format(p)] = agent.regret_nets[p].state_dict()
    cp["value_net"] = agent.value_net.state_dict()

    cp["avg_policy_t"] = agent.avg_policy_t
    cp["regret_t"] = agent.regret_t
    cp["value_t"] = agent.value_t

    cp_path = util.get_checkpoint_path(checkpoint_dir, agent.t)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(cp, cp_path)
    util.delete_old_checkpoints(checkpoint_dir)


def calc_nashconv(game, agent):
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


def train(cfg, agent):
    checkpoint_dir = os.path.join(cfg.run_dir, "checkpoint")
    load_checkpoint(agent, checkpoint_dir)

    agent.avg_policy_net.to(cfg.device)
    for p, _ in enumerate(agent.regret_nets):
        agent.regret_nets[p].to(cfg.device)
    agent.value_net.to(cfg.device)

    while True:
        agent.t += 1
        # _train_value(cfg, agent)
        _train_regret(cfg, agent)
        if agent.t % 20 == 0:
            _train_avg_policy(cfg, agent)
            _test_against(cfg, agent)

            conv = calc_nashconv(cfg.game, agent)
            cfg.summary_writer.add_scalar("nashconv", conv, agent.t)

        _save_checkpoint(checkpoint_dir, agent)
        cfg.summary_writer.flush()

        if agent.t % 20 == 0:
            logging.info("iteration %d", agent.t)


def _train_avg_policy(cfg, agent):
    num_epoch = 8
    epoch_steps = math.ceil(agent.cfg.avg_policy_batch_steps / num_epoch)
    dataloader_train = torch.utils.data.DataLoader(
            agent.avg_policy_buffer, batch_size=agent.cfg.avg_policy_batch_size, shuffle=True)
    optimizer = torch.optim.SGD(agent.avg_policy_net.parameters(), lr=agent.cfg.avg_policy_learning_rate)

    agent.avg_policy_net.linear.reset_parameters()
    # agent.avg_policy_net.trunk.linear0.reset_parameters()
    # agent.avg_policy_net.trunk.linear1.reset_parameters()

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


def _train_regret(cfg, agent):
    num_players = cfg.game.num_players()
    for player in range(num_players):
        for _ in range(agent.cfg.regret_traversals):
            state = cfg.game.new_initial_state()
            while not state.is_terminal():
                current_player = state.current_player()
                obs = state.information_state_tensor()
                mask = state.legal_actions_mask()

                policy = _match_regret(agent, current_player, obs, mask)
                if current_player == player:
                    # regret = _get_regret(agent, obs, policy)
                    regret = _get_regret_exact(agent, state)
                    sr = StateRegret(state=obs, regret=regret, mask=mask, t=agent.t)
                    agent.regret_buffers[player].add(sr)
                else:
                    behaviour = Behaviour(state=obs, policy=policy, t=agent.t)
                    agent.avg_policy_buffer.add(behaviour)

                if current_player == player:
                    sample_policy = mask / np.sum(mask)
                else:
                    sample_policy = policy
                action = np.random.choice(range(len(sample_policy)), p=sample_policy)
                state = state.child(action)

    for player, _ in enumerate(agent.regret_nets):
        num_epoch = 8
        epoch_steps = math.ceil(agent.cfg.regret_batch_steps / num_epoch)
        dataloader_train = torch.utils.data.DataLoader(
                agent.regret_buffers[player], batch_size=agent.cfg.regret_batch_size, shuffle=True)
        optimizer = torch.optim.SGD(agent.regret_nets[player].parameters(), lr=agent.cfg.regret_learning_rate)

        regret_net = agent.regret_nets[player]
        regret_net.linear.reset_parameters()
        # regret_net.trunk.linear0.reset_parameters()
        # regret_net.trunk.linear1.reset_parameters()

        for _ in range(num_epoch):
            agent.regret_t += 1
            metrics = {}

            regret_net.train()
            for _ in range(epoch_steps):
                batch = next(iter(dataloader_train))

                loss = _get_regret_loss(agent, player, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics = util.update_metric(metrics, "regret/train/loss", loss)

            for k, v in metrics.items():
                cfg.summary_writer.add_scalar(k, v.compute(), agent.regret_t)


def _train_value(cfg, agent):
    agent.value_buffer.clear()
    for _ in range(agent.cfg.value_traversals):
        state = cfg.game.new_initial_state()
        transitions = []
        while not state.is_terminal():
            player = state.current_player()
            obs = state.information_state_tensor()
            mask = state.legal_actions_mask()
            policy = _match_regret(agent, player, obs, mask)

            if np.random.uniform(0, 1) < 0.01:
                sample_policy = mask / np.sum(mask)
                a = np.random.choice(range(len(sample_policy)), p=sample_policy)
                importance = policy[a] / sample_policy[a]
            else:
                a = np.random.choice(range(len(policy)), p=policy)
                importance = 1

            state = state.child(a)
            reward = state.returns()[player]

            tn = Transition(player=player, observation=obs, importance=importance, action=a, reward=reward)
            transitions.append(tn)
        value = state.returns()

        last = transitions[len(transitions)-1]
        agent.value_buffer.add(StateActionValue(state=last.observation, action=last.action, value=value[last.player]))
        for i in range(len(transitions)-2, -1, -1):
            tn = transitions[i]
            next_tn = transitions[i+1]

            value[tn.player] += tn.reward
            value *= next_tn.importance

            sav = StateActionValue(
                    state=tn.observation, action=tn.action, value=value[tn.player])
            agent.value_buffer.add(sav)

    num_epoch = 8
    epoch_steps = math.ceil(agent.cfg.value_batch_steps / num_epoch)
    dataloader_train = torch.utils.data.DataLoader(
            agent.value_buffer, batch_size=agent.cfg.value_batch_size, shuffle=True)
    optimizer = torch.optim.SGD(agent.value_net.parameters(), lr=agent.cfg.value_learning_rate)

    agent.value_net.linear.reset_parameters()
    # agent.value_net.trunk.linear0.reset_parameters()
    # agent.value_net.trunk.linear1.reset_parameters()

    for _ in range(num_epoch):
        agent.value_t += 1
        metrics = {}

        agent.value_net.train()
        for _ in range(epoch_steps):
            batch = next(iter(dataloader_train))

            loss = _get_value_loss(agent, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = util.update_metric(metrics, "value/train/loss", loss)

        for k, v in metrics.items():
            cfg.summary_writer.add_scalar(k, v.compute(), agent.value_t)


def _get_avg_policy_loss(agent, batch):
    device = agent.avg_policy_net.linear.weight.device
    x = batch.state.to(torch.float32).to(device)
    y_policy = batch.policy.to(device)

    logits = agent.avg_policy_net(x)

    loss = torch.nn.functional.cross_entropy(logits, y_policy)

    # Linear CFR.
    weight = batch.t.to(device) / agent.t
    loss = torch.mul(loss, weight)

    return torch.mean(loss)


def _get_regret_loss(agent, player, batch):
    device = agent.avg_policy_net.linear.weight.device

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


def _get_regret(agent, obs, policy_np):
    device = agent.avg_policy_net.linear.weight.device
    with torch.no_grad():
        x = torch.from_numpy(obs).to(torch.float32).to(device)
        policy = torch.from_numpy(policy_np).to(device)

        action_value = agent.value_net(x)

        value = torch.sum(torch.mul(policy, action_value))
        regret = action_value - value
    return regret.cpu().numpy()


def _get_value_loss(agent, batch):
    device = agent.avg_policy_net.linear.weight.device

    x = batch.state.to(torch.float32).to(device)
    y_value = batch.value.to(device)

    action_values = agent.value_net(x)

    # Select values for each action.
    value = action_values[torch.arange(action_values.shape[0]), batch.action]

    loss = torch.pow(value - y_value, 2)
    return torch.mean(loss)


def _match_regret(agent, player, obs, mask_np):
    device = agent.avg_policy_net.linear.weight.device
    with torch.no_grad():
        x = torch.from_numpy(obs).to(torch.float32).to(device)
        mask = torch.from_numpy(mask_np).to(device)

        regrets = agent.regret_nets[player](x)

        regrets = torch.clamp(regrets, min=0)
        regrets = torch.mul(regrets, mask)

        summed = torch.sum(regrets)
        if summed > 1e-6:
            policy = regrets / summed
        else:
            policy = mask / torch.sum(mask)

        return policy.cpu().numpy()


def _get_value(agent, obs):
    with torch.no_grad():
        x = torch.from_numpy(obs).to(torch.float32)
        value = agent.value_net(x)
    return value.numpy()


def _get_regret_exact(agent, state):
    player = state.current_player()
    mask = state.legal_actions_mask()

    # Get value of children.
    children_values = np.zeros(mask.shape, dtype=float)
    for a, m in enumerate(mask):
        if m == 0:
            continue

        child = state.child(a)
        children_values[a] = _traverse_game_tree(player, agent, child)

    # Get policy.
    obs = state.information_state_tensor()
    policy = _match_regret(agent, player, obs, mask)

    value = 0
    for a, v in enumerate(children_values):
        value += policy[a] * v
    regret = children_values - value
    return regret


def _traverse_game_tree(player, agent, state):
    if state.is_terminal():
        return state.returns()[player]

    # Get value of children recursively.
    current_player = state.current_player()
    mask = state.legal_actions_mask()
    children_values = np.zeros(mask.shape, dtype=float)
    for a, m in enumerate(mask):
        if m == 0:
            continue

        child = state.child(a)
        children_values[a] = _traverse_game_tree(player, agent, child)

    # Get policy.
    obs = state.information_state_tensor()
    policy = _match_regret(agent, current_player, obs, mask)

    value = 0
    for a, v in enumerate(children_values):
        value += policy[a] * v
    return value


def _test_against(cfg, agent):
    num_trials = 100

    score = 0.0
    for i in range(num_trials):
        # Prepare game participants.
        values = np.zeros([cfg.game.num_players()], dtype=float)
        agent_idx = np.random.randint(len(values))
        agents = []
        for i in range(len(values)):
            if i == agent_idx:
                agents.append(agent)
            else:
                agents.append(cfg.test_agent)

        # Play game.
        state = cfg.game.new_initial_state()
        while not state.is_terminal():
            player = state.current_player()
            act = agents[player].get_action(state)

            state = state.child(act)
            values += state.returns()
        values += state.returns()

        # Add score.
        score += values[agent_idx]

    score /= num_trials
    cfg.summary_writer.add_scalar("score", score, agent.t)


class PolicyNet(torch.nn.Module):
    def __init__(self, trunk, action_dim):
        super().__init__()
        self.trunk = trunk
        self.linear = torch.nn.Linear(trunk.embedding_size, action_dim)

    def forward(self, observation):
        x = self.trunk(observation)
        return self.linear(x)


class Trunk(torch.nn.Module):
    def __init__(self, observation_dim, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.linear0 = torch.nn.Linear(observation_dim, embedding_size)
        self.relu0 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(embedding_size, embedding_size)
        self.relu1 = torch.nn.ReLU()

    def forward(self, observation):
        x = self.relu0(self.linear0(observation))
        x = x + self.relu1(self.linear1(x))
        return x


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
