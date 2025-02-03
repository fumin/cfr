import collections
import logging

import open_spiel
from open_spiel.python.algorithms import exploitability as _
from open_spiel.python.pytorch import deep_cfr as _
import numpy as np
import torch
import torch.utils.tensorboard as _

import util


MLP = open_spiel.python.pytorch.deep_cfr.MLP
DEVICE = "cpu"


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

            value_traversals=1024,
            value_batch_size=256,
            value_batch_steps=1024,
            value_learning_rate=1e-3,

            regret_traversals=1024,
            regret_batch_size=256,
            regret_batch_steps=375,
            regret_learning_rate=1e-3,

            avg_policy_batch_size=256,
            avg_policy_batch_steps=2500,
            avg_policy_learning_rate=1e-3,
            )
    return cfg


class Agent:
    def __init__(self, game, cfg):
        self.cfg = cfg
        self.t = 1
        history_dim = game.new_initial_state().history().shape[0]
        obs_dim = game.new_initial_state().information_state_tensor().shape[0]
        action_dim = game.num_distinct_actions()

        self._num_actions = action_dim

        self.avg_policy_buffer = ReservoirBuffer(cfg.memory_capacity)
        self.avg_policy_net = MLP(obs_dim, [64,], action_dim)

        self.regret_buffers = [ReservoirBuffer(cfg.memory_capacity) for _ in range(game.num_players())]
        self.regret_nets = [MLP(obs_dim, [64,], action_dim) for _ in range(game.num_players())]

        self.value_buffer = ReservoirBuffer(cfg.memory_capacity)
        self.value_net = MLP(history_dim, [128,], action_dim)
        self.value_dict = {}

        self.avg_policy_t = 0
        self.regret_t = 0
        self.value_t = 0

    def action_probabilities(self, obs, mask_np):
        # device = self.avg_policy_net.linear.weight.device
        device = torch.device(DEVICE)
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

            loss = _avg_policy_loss(agent, batch)
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
                regret = _get_regret(agent, state.history(), policy)
                # regret = _get_regret_exact(agent, state)
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
        _train_value(cfg, agent, player)
        _gather_regret_data(cfg.game, agent, player)

        num_epoch = 8
        epoch_steps = int(np.ceil(agent.cfg.regret_batch_steps / num_epoch))
        dataloader_train = torch.utils.data.DataLoader(agent.regret_buffers[player], batch_size=agent.cfg.regret_batch_size, shuffle=True)
        regret_net = agent.regret_nets[player]
        regret_net.reset()
        regret_net.to(torch.device(DEVICE))
        optimizer = torch.optim.Adam(regret_net.parameters(), lr=agent.cfg.regret_learning_rate)

        for _ in range(num_epoch):
            agent.regret_t += 1
            metrics = {}

            for _ in range(epoch_steps):
                batch = next(iter(dataloader_train))

                loss = _regret_loss(agent, player, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics = util.update_metric(metrics, "regret/train/loss", loss)

            for k, v in metrics.items():
                cfg.summary_writer.add_scalar(k, v.compute(), agent.regret_t)


def _gather_value_data(game, agent, player):
    agent.value_buffer.clear()
    for _ in range(agent.cfg.value_traversals):
        state = game.new_initial_state()
        transitions = []
        tn_states_m1 = []
        tn_states = []
        while not state.is_terminal():
            tn_states_m1.append(game.from_spiel(state._state.clone()))

            # Get policy.
            history = state.history()
            current_player = state.current_player()
            obs = state.information_state_tensor()
            mask = state.legal_actions_mask()
            policy = _match_regret(agent, current_player, obs, mask)

            # Sample action.
            if np.random.uniform(0, 1) < 1:#0.01:
                sample_policy = mask / np.sum(mask)
                action = np.random.choice(range(len(sample_policy)), p=sample_policy)
                importance = policy[action] / sample_policy[action]
            else:
                action = np.random.choice(range(len(policy)), p=policy)
                importance = 1

            # Get returns.
            state = state.child(action)
            returns = state.returns()

            # Add transition.
            tn = Transition(player=current_player, history=history, importance=importance, action=action, returns=returns)
            transitions.append(tn)
            tn_states.append(game.from_spiel(state._state.clone()))

        value = np.zeros(transitions[0].returns.shape, dtype=float)
        for i in range(len(transitions)-1, -1, -1):
            tn = transitions[i]

            value += tn.returns
            agent.value_buffer.add(StateActionValue(state=tn.history, action=tn.action, value=value[tn.player]))

            value *= tn.importance


def _train_value(cfg, agent, player):
    _gather_value_data(cfg.game, agent, player)

    agent.value_dict = {}
    for i in range(len(agent.value_buffer)):
        sav = agent.value_buffer[i]

        key = str(sav.state) + str(sav.action)
        if not (key in agent.value_dict):
            agent.value_dict[key] = []

        agent.value_dict[key].append(sav.value)

    for k, v in agent.value_dict.items():
        agent.value_dict[k] = np.mean(v)
    
    num_epoch = 8
    epoch_steps = int(np.ceil(agent.cfg.value_batch_steps / num_epoch))
    dataloader_train = torch.utils.data.DataLoader(agent.value_buffer, batch_size=agent.cfg.value_batch_size, shuffle=True)
    agent.value_net.reset()
    agent.value_net.to(torch.device(DEVICE))
    optimizer = torch.optim.Adam(agent.value_net.parameters(), lr=agent.cfg.value_learning_rate)
    
    for _ in range(num_epoch):
        agent.value_t += 1
        metrics = {}
    
        for _ in range(epoch_steps):
            batch = next(iter(dataloader_train))
    
            loss = _get_value_loss(agent, batch)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            metrics = util.update_metric(metrics, "value/train/loss", loss)
    
        for k, v in metrics.items():
            cfg.summary_writer.add_scalar(k, v.compute(), agent.value_t)


def _avg_policy_loss(agent, batch):
    # device = agent.avg_policy_net.linear.weight.device
    device = torch.device(DEVICE)
    x = batch.state.to(torch.float32).to(device)
    y_policy = batch.policy.to(device)

    logits = agent.avg_policy_net(x)

    loss = torch.nn.functional.cross_entropy(logits, y_policy)

    # Linear CFR.
    weight = batch.t.to(device) / agent.t
    loss = torch.mul(loss, weight)

    return torch.mean(loss)


def _regret_loss(agent, player, batch):
    # device = agent.avg_policy_net.linear.weight.device
    device = torch.device(DEVICE)

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
    device = torch.device(DEVICE)
    with torch.no_grad():
        x = torch.from_numpy(obs).to(torch.float32).to(device)
        # logging.info("x.device %s", x.device)
        # for i, l in enumerate(agent.regret_nets[player]._layers):
        #     logging.info("%d %s %s", i, l._weight.device, l._bias.device)
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


def _get_regret(agent, history, policy_np):
    # device = agent.avg_policy_net.linear.weight.device
    device = torch.device(DEVICE)
    with torch.no_grad():
        x = torch.from_numpy(history).to(torch.float32).to(device)
        policy = torch.from_numpy(policy_np).to(device)

        children_values = agent.value_net(x)

        value = torch.sum(policy * children_values)
        regret = children_values - value
    return regret.cpu().numpy()


def _get_regret_exact(agent, state):
    player = state.current_player()

    # Get value of children.
    mask = state.legal_actions_mask()
    children_values = np.zeros(mask.shape, dtype=float)
    for a, m in enumerate(mask):
        if m == 1:
            child = state.child(a)
            children_values[a] = _value_exact(player, agent, child)
            good = children_values[a]

            ss = np.concatenate([[player], state.history()])
            ss = state.history()
            key = str(ss) + str(a)
            if not (key in agent.value_dict):
                logging.info("%s", agent.value_dict)
                raise ValueError(key)
            children_values[a] = agent.value_dict[key]
            # logging.info("%s good %f nn %f player %d \"%s\" action %d", np.abs(good-children_values[a]), good, children_values[a], player, state._state, a)
            if False and np.abs(good-children_values[a]) > 1e-6 and str(state._state) == "1 0 p":
                logging.info("child \"%s\"", child._state)
                logging.info("key \"%s\"", key)
                logging.info("%s good %f bad %f player %d action %d \"%s\"", np.abs(good-children_values[a])<1e-6, good, children_values[a], player, a, state._state)
                raise ValueError("???")

            current_player = state.current_player()
            history = state.history()
            obs = state.information_state_tensor()
            mask = state.legal_actions_mask()
            policy_np = _match_regret(agent, current_player, obs, mask)
            device = torch.device(DEVICE)
            with torch.no_grad():
                x = torch.from_numpy(history).to(torch.float32).to(device)
                policy = torch.from_numpy(policy_np).to(device)

                children_values_nn = agent.value_net(x)
            nnvalues = children_values_nn.cpu().numpy()
            children_values[a] = nnvalues[a]
            # logging.info("%s good %f nn %f player %d \"%s\" action %d", np.abs(good-children_values[a]), good, children_values[a], player, state._state, a)

    # Get policy.
    obs = state.information_state_tensor()
    policy = _match_regret(agent, player, obs, mask)

    value = np.sum(policy * children_values)
    regret = children_values - value
    return regret


def _get_value_loss(agent, batch):
    # device = agent.avg_policy_net.linear.weight.device
    device = torch.device(DEVICE)

    x = batch.state.to(torch.float32).to(device)
    y_value = batch.value.to(device)

    # logging.info("x.device %s", x.device)
    # for i, layer in enumerate(agent.value_net._layers):
    #     logging.info("%d %s %s", i, layer._weight.device, layer._bias.device)
    action_values = agent.value_net(x)

    # Select values for each action.
    batch_size = action_values.shape[0]
    value = action_values[torch.arange(batch_size), batch.action]

    loss = torch.pow(value - y_value, 2)
    return torch.mean(loss)


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
    agent.avg_policy_net.to(torch.device(DEVICE))
    for net in agent.regret_nets:
        net.to(torch.device(DEVICE))
    agent.value_net.to(torch.device(DEVICE))

    for _ in range(999999):
        _train_regret(cfg, agent)
        agent.t += 1

        if agent.t % 20 == 0:
            _train_avg_policy(cfg, agent)
            _calc_nashconv(cfg.game, agent)


Transition = collections.namedtuple("Transition", ["player", "history", "importance", "action", "returns"])
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
