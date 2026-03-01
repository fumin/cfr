import collections
import logging

import open_spiel
from open_spiel.python.algorithms import exploitability as _
from open_spiel.python.pytorch import deep_cfr as _
import numpy as np
import torch
import torch.utils.tensorboard as _
import torcheval
from torcheval import metrics as _


class Config:
    def __init__(self):
        # Kuhn poker, 100 iters 0.046, 200 iters 0.01.
        self.value_traversals = 512
        self.value_exploration = 0.1
        self.value_memory_capacity = 1e6
        self.value_net = [64]
        self.value_batch_size = 256
        self.value_batch_steps = 512
        self.value_learning_rate = 1e-3

        self.regret_traversals = 1024
        self.regret_memory_capacity = 1e6
        self.regret_net = [64]
        self.regret_batch_size = 256
        self.regret_batch_steps = 375
        self.regret_learning_rate = 1e-3

        self.avg_policy_memory_capacity = 1e6
        self.avg_policy_net = [64]
        self.avg_policy_batch_size = 256
        self.avg_policy_batch_steps = 2500
        self.avg_policy_learning_rate = 1e-3


class Agent:
    def __init__(self, game, cfg):
        self.cfg = cfg
        self.t = 1

        state = game.new_initial_state()
        history_dim = _state_history(game.num_players(), state).shape[0]
        obs_dim = game.information_state_tensor_size()
        action_dim = game.num_distinct_actions()

        self.avg_policy_buffer = ReservoirBuffer(cfg.avg_policy_memory_capacity)
        MLP = open_spiel.python.pytorch.deep_cfr.MLP
        self.avg_policy_net = MLP(obs_dim, cfg.avg_policy_net, action_dim)

        self.regret_buffers = [ReservoirBuffer(cfg.regret_memory_capacity) for _ in range(game.num_players())]
        self.regret_nets = [MLP(obs_dim, cfg.regret_net, action_dim) for _ in range(game.num_players())]

        self.value_buffers = [ReservoirBuffer(cfg.value_memory_capacity) for _ in range(game.num_players())]
        self.value_nets = [MLP(history_dim, cfg.value_net, 1) for _ in range(game.num_players())]
        self.value_maps = [{} for _ in range(game.num_players())]

        self.num_touched = 0
        self.avg_policy_t = 0
        self.regret_t = 0
        self.value_t = 0

    def action_probabilities(self, obs, mask_np):
        device = _mlp_device(self.avg_policy_net)
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
    dataloader_train = torch.utils.data.DataLoader(agent.avg_policy_buffer, batch_size=agent.cfg.avg_policy_batch_size, shuffle=True)
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

            metrics = _update_metric(metrics, "avg_policy/train/loss", loss)

        for k, v in metrics.items():
            cfg.summary_writer.add_scalar(k, v.compute(), agent.avg_policy_t)


def _gather_regret_data(game, agent, player):
    for _ in range(agent.cfg.regret_traversals):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                a = np.random.choice(actions, p=probs)
                state.apply_action(a)
                continue

            # Get policy.
            current_player = state.current_player()
            obs = np.array(state.information_state_tensor(), dtype=float)
            mask = np.array(state.legal_actions_mask(), dtype=int)
            policy = _match_regret(agent.regret_nets[current_player], obs, mask)

            # Add data to buffer.
            if current_player == player:
                regret = _get_regret(agent, state, policy, game.num_players())
                # regret = _get_regret_exact(agent, state, policy)
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

                metrics = _update_metric(metrics, "regret/train/loss", loss)

            for k, v in metrics.items():
                cfg.summary_writer.add_scalar(k, v.compute(), agent.regret_t)


def _gather_value_data(game, agent, player):
    value_buffer = agent.value_buffers[player]
    value_buffer.clear()
    for _ in range(agent.cfg.value_traversals):
        state = game.new_initial_state()
        agent.num_touched += 1
        transitions = []
        while True:
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                a = np.random.choice(actions, p=probs)
                state.apply_action(a)
                continue

            action, importance = -1, 1
            if not state.is_terminal():
                # Get policy.
                obs = np.array(state.information_state_tensor(), dtype=float)
                mask = np.array(state.legal_actions_mask(), dtype=int)
                policy = _match_regret(agent.regret_nets[state.current_player()], obs, mask)

                # Sample action.
                # epsilon = np.clip(agent.cfg.value_exploration / np.log(1+agent.t), a_min=0.01, a_max=1)
                epsilon = agent.cfg.value_exploration
                uniform = mask / np.sum(mask)
                sample_policy = epsilon*uniform + (1-epsilon)*policy
                action = np.random.choice(range(len(sample_policy)), p=sample_policy)
                importance = policy[action] / sample_policy[action]

            # Add transition.
            history = _state_history(game.num_players(), state)
            returns = np.array(state.returns(), dtype=float)
            tn = Transition(history=history, importance=importance, action=action, returns=returns)
            transitions.append(tn)

            if state.is_terminal():
                break
            state = state.child(action)
            agent.num_touched += 1

        value = np.zeros(transitions[0].returns.shape, dtype=float)
        for i in range(len(transitions)-1, -1, -1):
            tn = transitions[i]

            value = tn.importance * (tn.returns + value)
            value_buffer.add(StateActionValue(state=tn.history, action=tn.action, value=value[player]))


def _gather_value_data_exact(game, agent, player):
    value_buffer = agent.value_buffers[player]
    value_buffer.clear()
    def callback(state, value):
        history = _state_history(game.num_players(), state)
        value_buffer.add(StateActionValue(state=history, action=-1, value=value))
        agent.num_touched += 1

    # Kuhn poker has 3 chance nodes.
    # A population of 60 seems to be sufficient to sample all of them.
    for _ in range(60):
        state = game.new_initial_state()
        _value_exact(agent, player, state, callback)


def _train_value(cfg, agent, player):
    # _gather_value_data_exact(cfg.game, agent, player)
    _gather_value_data(cfg.game, agent, player)

    # value_map = agent.value_maps[player]
    # logging.info("%d player %d map %d", agent.t, player, len(value_map))
    # value_map.clear()
    # for sav in agent.value_buffers[player]:
    #     k = str(sav.state)
    #     value_map[k] = sav.value

    # dataset = {}
    # dataset["x"] = np.zeros([len(value_map), agent.value_buffers[player][0].state.shape[0]], dtype=float)
    # dataset["y"] = np.zeros([len(value_map)], dtype=float)
    # kss = {}
    # for sav in agent.value_buffers[player]:
    #     k = str(sav.state)
    #     if k not in kss:
    #         kss[k] = 1
    #         dataset["x"][len(kss)-1, :] = sav.state
    #         dataset["y"][len(kss)-1] = sav.value
    # fpath = "data_{}_{}.pth".format(agent.t, player)
    # torch.save(dataset, fpath)

    # return

    num_epoch = 8
    epoch_steps = int(np.ceil(agent.cfg.value_batch_steps / num_epoch))
    dataloader_train = torch.utils.data.DataLoader(agent.value_buffers[player], batch_size=agent.cfg.value_batch_size, shuffle=True)
    value_net = agent.value_nets[player]
    value_net.reset()
    optimizer = torch.optim.Adam(value_net.parameters(), lr=agent.cfg.value_learning_rate)

    for _ in range(num_epoch):
        agent.value_t += 1
        metrics = {}

        agent.value_nets[player].train()
        for _ in range(epoch_steps):
            batch = next(iter(dataloader_train))

            loss = _get_value_loss(agent, player, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = _update_metric(metrics, "value/train/loss", loss)

        for k, v in metrics.items():
            cfg.summary_writer.add_scalar(k, v.compute(), agent.value_t)


def _get_avg_policy_loss(agent, batch):
    device = _mlp_device(agent.avg_policy_net)
    x = batch.state.to(torch.float32).to(device)
    y_policy = batch.policy.to(device)

    logits = agent.avg_policy_net(x)

    loss = torch.nn.functional.cross_entropy(logits, y_policy)

    # Linear CFR.
    weight = batch.t.to(device) / agent.t
    loss = torch.mul(loss, weight)

    return torch.mean(loss)


def _get_regret_loss(agent, player, batch):
    device = _mlp_device(agent.regret_nets[player])

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


def _get_value_loss(agent, player, batch):
    device = _mlp_device(agent.value_nets[player])

    x = batch.state.to(torch.float32).to(device)
    y_value = batch.value.to(device)

    value = agent.value_nets[player](x)
    value = torch.squeeze(value, dim=[1])

    loss = torch.pow(value - y_value, 2)
    return torch.mean(loss)


def _match_regret(net, obs, mask_np):
    device = _mlp_device(net)
    with torch.no_grad():
        x = torch.from_numpy(obs).to(torch.float32).to(device)
        regrets = net(x)
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


def _get_regret(agent, state, policy, num_players):
    player = state.current_player()
    device = _mlp_device(agent.value_nets[player])

    mask = state.legal_actions_mask()
    children_values = np.zeros(len(mask), dtype=float)
    for a, m in enumerate(mask):
        if m == 1:
            child = state.child(a)

            # value_map = agent.value_maps[player]
            # k = str(child.history())
            # if k not in value_map:
            #     raise Exception("%d %s".format(len(value_map), k))
            # v = value_map[k]
            # v_ok = _value_exact(agent, player, child, None)
            # if v != v_ok:
            #     raise Exception("{} != {}".format(v, v_ok))
            # children_values[a] = v
            # continue

            with torch.no_grad():
                history = _state_history(num_players, child)
                x = torch.from_numpy(history).to(torch.float32).to(device)
                children_values[a] = agent.value_nets[player](x)

    value = np.sum(policy * children_values)
    regret = children_values - value
    return regret


def _get_regret_exact(agent, state, policy):
    player = state.current_player()

    # Get value of children.
    mask = state.legal_actions_mask()
    children_values = np.zeros(len(mask), dtype=float)
    for a, m in enumerate(mask):
        if m == 1:
            child = state.child(a)
            children_values[a] = _value_exact(agent, player, child, None)

    value = np.sum(policy * children_values)
    regret = children_values - value
    return regret


def _value_exact(agent, player, state, callback):
    if state.is_terminal():
        if callback:
            callback(state, state.returns()[player])
        return state.returns()[player]

    # Get children values recursively.
    mask = state.legal_actions_mask()
    children_values = np.zeros(mask.shape, dtype=float)
    for a, m in enumerate(mask):
        if m == 1:
            child = state.child(a)
            children_values[a] = _value_exact(agent, player, child, callback)

    # Get policy.
    current_player = state.current_player()
    obs = np.array(state.information_state_tensor(), dtype=float)
    policy = _match_regret(agent.regret_nets[current_player], obs, mask)

    value = np.sum(policy * children_values)

    if callback:
        callback(state, value)
    return value


def _calc_nashconv(game, agent):
    def action_probabilities(state):
        obs = np.array(state.information_state_tensor(), dtype=float)
        mask = np.array(state.legal_actions_mask(), dtype=int)
        probs = agent.action_probabilities(obs, mask)
        return {a: probs[a] for a in state.legal_actions()}

    policy = open_spiel.python.policy.tabular_policy_from_callable(game, action_probabilities)
    conv = open_spiel.python.algorithms.exploitability.nash_conv(game, policy)
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
    agent.avg_policy_net.to(cfg.device)
    for i in range(len(agent.regret_nets)):
        agent.regret_nets[i].to(cfg.device)
        agent.value_nets[i].to(cfg.device)

    for _ in range(999999):
        _train_regret(cfg, agent)
        agent.t += 1

        if agent.t % 20 == 0:
            _train_avg_policy(cfg, agent)
            _calc_nashconv(cfg.game, agent)


Transition = collections.namedtuple("Transition", ["history", "importance", "action", "returns"])
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


def _mlp_device(net):
    return net._layers[0]._weight.device


def _state_history(num_players, state):
    history = []
    for p in range(num_players):
        history += state.information_state_tensor(p)
    return np.array(history, dtype=float)


def _update_metric(metrics, k, v):
    if not(k in metrics):
        metrics[k] = torcheval.metrics.Mean().to(v.device)
    metrics[k].update(v)
    return metrics
