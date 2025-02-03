import numpy as np
import pyspiel


class Game:
    def __init__(self, spiel_game):
        self._game = spiel_game
    
    def new_initial_state(self):
        state = self._game.new_initial_state()
        state = child_all_chance_nodes(state)
        return self.from_spiel(state)

    def num_distinct_actions(self):
        return self._game.num_distinct_actions()

    def num_players(self):
        return self._game.num_players()

    def from_spiel(self, spiel_state):
        return State(self._game.get_type(), self.num_players(), spiel_state)


class State:
    def __init__(self, game_type, num_players, spiel_state):
        self._game_type = game_type
        self._num_players = num_players
        self._state = spiel_state
    
    def child(self, action):
        spiel_state = self._state.child(action)
        spiel_state = child_all_chance_nodes(spiel_state)
        return State(self._game_type, self._num_players, spiel_state)

    def history(self):
        if self._game_type.provides_information_state_tensor:
            tsfn = self._state.information_state_tensor
        else:
            tsfn = self._state.observation_tensor

        history = []
        for p in range(self._num_players):
            history += tsfn(p)
        return np.array(history, dtype=float)

    def information_state_tensor(self):
        player = self.current_player()

        if self._game_type.provides_information_state_tensor:
            obs = self._state.information_state_tensor(player)
        else:
            obs = self._state.observation_tensor(player)

        ts = [player] + obs
        return np.array(ts, dtype=float)
    
    def legal_actions_mask(self):
        player = self.current_player()
        ts = self._state.legal_actions_mask(player)
        return np.array(ts, dtype=int)
    
    def returns(self):
        ts = self._state.returns()
        return np.array(ts, dtype=float)
    
    def is_terminal(self):
        return self._state.is_terminal()

    def current_player(self):
        return self._state.current_player()

    # remove me, only for depfr
    def legal_actions(self, player=None):
        if player == None:
            return self._state.legal_actions()
        return self._state.legal_actions(player)


def child_all_chance_nodes(state):
    while state.is_chance_node():
        action_list, prob_list = zip(*state.chance_outcomes())
        action = np.random.choice(action_list, p=prob_list)
        state = state.child(action)
    return state
