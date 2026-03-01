import collections
import logging
import os

import open_spiel
from open_spiel.python.algorithms import exploitability as _
from open_spiel.python.pytorch import deep_cfr as _
import numpy as np
import torch
import torch.utils.tensorboard as _

import util


class Config:
    def __init__(self):
        # Default parameters for kuhn poker, which should result in an exploitability (nashconv) of less than 0.03 in roughly 100 iterations.
        self.memory_capacity = int(2.5e5)

        self.value_traversals = 1024
        self.value_exploration = 1
        self.value_net = [128]
        self.value_batch_size = 256
        self.value_batch_steps = 1024
        self.value_learning_rate = 1e-3

        self.regret_traversals = 1024
        self.regret_net = [64]
        self.regret_batch_size = 256
        self.regret_batch_steps = 375
        self.regret_learning_rate = 1e-3

        self.avg_policy_net = [64]
        self.avg_policy_batch_size = 256
        self.avg_policy_batch_steps = 2500
        self.avg_policy_learning_rate = 1e-3


class Agent:
    def __init__(self, game, cfg):
        self.cfg = cfg
        self.t = 0

        state = game.new_initial_state()
        history_dim = _player_history(0, state.history()).shape[0]
        obs_dim = state.information_state_tensor().shape[0]
        action_dim = game.num_distinct_actions()

        self.value_buffer = ReservoirBuffer(cfg.memory_capacity)
        self.value_net = MLP(history_dim, cfg.value_net, action_dim)

        self.regret_buffers = [ReservoirBuffer(cfg.memory_capacity) for _ in range(game.num_players())]
        self.regret_nets = [MLP(obs_dim, cfg.regret_net, action_dim) for _ in range(game.num_players())]

        self.avg_policy_buffer = ReservoirBuffer(cfg.memory_capacity)
        self.avg_policy_net = MLP(obs_dim, cfg.avg_policy_net, action_dim)

        self.num_touched = 0
        self.avg_policy_t = 0
        self.regret_t = 0
        self.value_t = 0

    def action_probabilities(self, obs, mask_np):
        device = self.avg_policy_net.device()
        with torch.no_grad():
            x = torch.from_numpy(obs).to(torch.float32).to(device)
            mask = torch.from_numpy(mask_np).to(device)

            logits = self.avg_policy_net(x)
            probs = torch.nn.functional.softmax(logits, dim=0)

            probs = torch.mul(probs, mask)
            probs = probs / torch.sum(probs)

        return probs.cpu().numpy()

    def get_action(self, state):
        obs = state.information_state_tensor()
        mask = state.legal_actions_mask()
        policy = self.action_probabilities(obs, mask)
        action = np.random.choice(range(len(policy)), p=policy)
        return action


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
        agent.num_touched += 1
        my_sample_reach = 1
        while not state.is_terminal():
            # Get policy.
            current_player = state.current_player()
            regret_net = agent.regret_nets[current_player]
            obs = state.information_state_tensor()
            mask = state.legal_actions_mask()
            policy = _match_regret(regret_net, obs, mask)

            # Get action.
            if current_player == player:
                sample_policy = mask / np.sum(mask)
            else:
                sample_policy = policy
            action = np.random.choice(range(len(sample_policy)), p=sample_policy)

            # Add data to buffer.
            if current_player == player:
                p_history = _player_history(current_player, state.history())
                regret = _get_regret(agent, p_history, policy)

                # Encourage learning leaf regrets.
                regret *= min(1./my_sample_reach, 100)
                my_sample_reach *= sample_policy[action]

                sr = StateRegret(state=obs, regret=regret, mask=mask, t=agent.t)
                agent.regret_buffers[player].add(sr)
            else:
                behaviour = Behaviour(state=obs, policy=policy, t=agent.t)
                agent.avg_policy_buffer.add(behaviour)

            # Update state.
            state = state.child(action)
            agent.num_touched += 1


def _train_regret(cfg, agent):
    for player in range(cfg.game.num_players()):
        _train_value(cfg, agent, player)
        _gather_regret_data(cfg.game, agent, player)

        num_epoch = 8
        epoch_steps = int(np.ceil(agent.cfg.regret_batch_steps / num_epoch))
        dataloader_train = torch.utils.data.DataLoader(agent.regret_buffers[player], batch_size=agent.cfg.regret_batch_size, shuffle=True)
        regret_net = agent.regret_nets[player]
        regret_net.reset_parameters()
        optimizer = torch.optim.Adam(regret_net.parameters(), lr=agent.cfg.regret_learning_rate)

        for _ in range(num_epoch):
            agent.regret_t += 1
            metrics = {}

            regret_net.train()
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
        agent.num_touched += 1
        transitions = []
        while not state.is_terminal():
            # Get policy.
            history = state.history()
            current_player = state.current_player()
            regret_net = agent.regret_nets[current_player]
            obs = state.information_state_tensor()
            mask = state.legal_actions_mask()
            policy = _match_regret(regret_net, obs, mask)

            # Sample action.
            epsilon = np.clip(agent.cfg.value_exploration / np.log(1+agent.t), a_min=0.01, a_max=1)
            # epsilon = agent.cfg.value_exploration
            if np.random.uniform(0, 1) < epsilon:
                sample_policy = mask / np.sum(mask)
                action = np.random.choice(range(len(sample_policy)), p=sample_policy)
                importance = policy[action] / sample_policy[action]
            else:
                action = np.random.choice(range(len(policy)), p=policy)
                importance = 1

            # Get returns.
            state = state.child(action)
            agent.num_touched += 1
            returns = state.returns()

            # Add transition.
            tn = Transition(history=history, importance=importance, action=action, returns=returns)
            transitions.append(tn)

        value = np.zeros(transitions[0].returns.shape, dtype=float)
        for i in range(len(transitions)-1, -1, -1):
            tn = transitions[i]

            value += tn.returns
            p_history = _player_history(player, tn.history)
            agent.value_buffer.add(StateActionValue(state=p_history, action=tn.action, value=value[player]))

            value *= tn.importance


def _train_value(cfg, agent, player):
    _gather_value_data(cfg.game, agent, player)

    num_epoch = 8
    epoch_steps = int(np.ceil(agent.cfg.value_batch_steps / num_epoch))
    dataloader_train = torch.utils.data.DataLoader(agent.value_buffer, batch_size=agent.cfg.value_batch_size, shuffle=True)
    agent.value_net.reset_parameters()
    optimizer = torch.optim.Adam(agent.value_net.parameters(), lr=agent.cfg.value_learning_rate)
    
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


def _avg_policy_loss(agent, batch):
    device = agent.avg_policy_net.device()
    x = batch.state.to(torch.float32).to(device)
    y_policy = batch.policy.to(device)

    logits = agent.avg_policy_net(x)

    loss = torch.nn.functional.cross_entropy(logits, y_policy)

    # Linear CFR.
    weight = batch.t.to(device) / agent.t
    loss = torch.mul(loss, weight)

    return torch.mean(loss)


def _regret_loss(agent, player, batch):
    device = agent.regret_nets[player].device()

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


def _match_regret(regret_net, obs, mask_np):
    device = regret_net.device()
    with torch.no_grad():
        x = torch.from_numpy(obs).to(torch.float32).to(device)
        regrets = regret_net(x)
    raw_regrets = regrets.cpu().numpy()

    regrets = np.clip(raw_regrets, a_min=0, a_max=None)
    regrets = regrets * mask_np
    summed = np.sum(regrets)
    if summed > 1e-6:
        return regrets / summed

    # Just use the best regret.
    max_id, max_regret = -1, np.finfo(np.float32).min
    for i, m in enumerate(mask_np):
        if m == 1 and raw_regrets[i] > max_regret:
            max_id, max_regret = i, raw_regrets[i]
    policy = np.zeros(regrets.shape, dtype=regrets.dtype)
    policy[max_id] = 1
    return policy


def _get_regret(agent, history, policy_np):
    device = agent.value_net.device()
    with torch.no_grad():
        x = torch.from_numpy(history).to(torch.float32).to(device)
        policy = torch.from_numpy(policy_np).to(device)

        children_values = agent.value_net(x)

        value = torch.sum(policy * children_values)
        regret = children_values - value
    return regret.cpu().numpy()


def _get_value_loss(agent, batch):
    device = agent.value_net.device()

    x = batch.state.to(torch.float32).to(device)
    y_value = batch.value.to(device)

    action_values = agent.value_net(x)

    # Select values for each action.
    batch_size = action_values.shape[0]
    value = action_values[torch.arange(batch_size), batch.action]

    loss = torch.pow(value - y_value, 2)
    return torch.mean(loss)


def load_checkpoint(agent, checkpoint_dir):
    torch.serialization.add_safe_globals([Behaviour, StateRegret])

    cp, cp_path = util.load_checkpoint(checkpoint_dir)
    if not cp:
        logging.info("no checkpoint")
        return

    agent.t = cp["t"]
    agent.avg_policy_net.load_state_dict(cp["avg_policy_net"])
    for p, _ in enumerate(agent.regret_nets):
        agent.regret_nets[p].load_state_dict(cp["regret_net_{}".format(p)])
    agent.value_net.load_state_dict(cp["value_net"])

    agent.num_touched = cp["num_touched"]
    agent.avg_policy_t = cp["avg_policy_t"]
    agent.regret_t = cp["regret_t"]
    agent.value_t = cp["value_t"]

    agent.avg_policy_buffer.load_state_dict(cp["avg_policy_buffer"])
    capacity = agent.avg_policy_buffer._buf._reservoir_buffer_capacity
    agent.avg_policy_buffer._buf._data = agent.avg_policy_buffer._buf._data[:capacity]
    logging.info("loaded avg_policy_buffer %d", len(agent.avg_policy_buffer))
    for p in range(len(agent.regret_buffers)):
        agent.regret_buffers[p].load_state_dict(cp["regret_buffer_{}".format(p)])
        agent.regret_buffers[p]._buf._data = agent.regret_buffers[p]._buf._data[:capacity]
        logging.info("loaded regret_buffer[%d] %d", p, len(agent.regret_buffers[p]))

    logging.info("loaded checkpoint %s", cp_path)


def _save_checkpoint(checkpoint_dir, agent):
    cp = {}
    cp["t"] = agent.t
    cp["avg_policy_net"] = agent.avg_policy_net.state_dict()
    for p, _ in enumerate(agent.regret_nets):
        cp["regret_net_{}".format(p)] = agent.regret_nets[p].state_dict()
    cp["value_net"] = agent.value_net.state_dict()

    cp["num_touched"] = agent.num_touched
    cp["avg_policy_t"] = agent.avg_policy_t
    cp["regret_t"] = agent.regret_t
    cp["value_t"] = agent.value_t

    cp["avg_policy_buffer"] = agent.avg_policy_buffer.state_dict()
    for p in range(len(agent.regret_buffers)):
        cp["regret_buffer_{}".format(p)] = agent.regret_buffers[p].state_dict()

    cp_path = util.get_checkpoint_path(checkpoint_dir, agent.t)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(cp, cp_path, _use_new_zipfile_serialization=False)
    util.delete_old_checkpoints(checkpoint_dir)


class TrainConfig:
    def __init__(self):
        self.run_dir = ""
        self.device_name = ""
        self.game = None

        self.test_every = 20
        self.test_agent = None
        self.nashconv = False

        # Derived fields.
        self.summary_writer = None
        self.device = None

    def setup(self):
        self.summary_writer = torch.utils.tensorboard.SummaryWriter(self.run_dir)
        self.device = torch.device(self.device_name)


def train(cfg, agent):
    checkpoint_dir = os.path.join(cfg.run_dir, "checkpoint")
    load_checkpoint(agent, checkpoint_dir)

    agent.avg_policy_net.to(cfg.device)
    for net in agent.regret_nets:
        net.to(cfg.device)
    agent.value_net.to(cfg.device)

    for _ in range(999999):
        agent.t += 1
        _train_regret(cfg, agent)

        if agent.t % cfg.test_every == 0 or agent.t == 1:
            _train_avg_policy(cfg, agent)

            _test_against(cfg, agent)
            _calc_nashconv(cfg, agent)

            _save_checkpoint(checkpoint_dir, agent)
            cfg.summary_writer.flush()


def _test_against(cfg, agent):
    num_trials = 1000

    score = 0.0
    for i in range(num_trials):
        # Prepare game participants.
        agents = [cfg.test_agent for i in range(cfg.game.num_players())]
        agent_idx = np.random.randint(len(agents))
        agents[agent_idx] = agent
        values = np.zeros([len(agents)], dtype=float)

        # Play game.
        state = cfg.game.new_initial_state()
        while not state.is_terminal():
            player = state.current_player()
            act = agents[player].get_action(state)

            state = state.child(act)
            values += state.returns()
        values += state.returns()

        # Add score.
        if values[agent_idx] > 0:
            score += 1
    score /= num_trials

    cfg.summary_writer.add_scalar("score", score, agent.num_touched)
    logging.info("%d states %d score %f", agent.t, agent.num_touched, score)


def _calc_nashconv(cfg, agent):
    if not cfg.nashconv:
        return

    def action_probabilities(spiel_state):
        state = cfg.game.from_spiel(spiel_state)
        obs = state.information_state_tensor()
        mask = state.legal_actions_mask()
        probs = agent.action_probabilities(obs, mask)
        return {a: probs[a] for a in spiel_state.legal_actions()}

    policy = open_spiel.python.policy.tabular_policy_from_callable(cfg.game._game, action_probabilities)
    conv = open_spiel.python.algorithms.exploitability.nash_conv(cfg.game._game, policy)

    cfg.summary_writer.add_scalar("nashconv", conv, agent.t)
    logging.info("iteration %d nashconv %f", agent.t, conv)


Transition = collections.namedtuple("Transition", ["history", "importance", "action", "returns"])
StateActionValue = collections.namedtuple(
        "StateActionValue", ["state", "action", "value"])
StateRegret = collections.namedtuple(
        "StateRegret", ["state", "regret", "mask", "t"])
Behaviour = collections.namedtuple(
        "Behaviour", ["state", "policy", "t"])


def _player_history(player, history):
    return np.concatenate([[player], history])


class ReservoirBuffer:
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

    def state_dict(self):
        d = {}
        d["data"] = self._buf._data
        d["add_calls"] = self._buf._add_calls
        return d

    def load_state_dict(self, d):
        self._buf._data = d["data"]
        self._buf._add_calls = d["add_calls"]


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self._mlp = open_spiel.python.pytorch.deep_cfr.MLP(input_size, hidden_sizes, output_size)

    def forward(self, x):
        return self._mlp(x)

    def reset_parameters(self):
        device = self.device()
        self._mlp.reset()
        self._mlp.to(device)

    def device(self):
        return self._mlp._layers[0]._weight.device
