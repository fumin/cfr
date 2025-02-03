import numpy as np

import util


class Human:
    def __init__(self):
        pass

    def get_action(self, state):
        mask = state.legal_actions_mask()
        while True:
            inp = input("your turn:")
            a, err = util.atoi(inp)
            if err:
                continue
            if a >= 0 and a < len(mask) and mask[a] == 1:
                return a


class Random:
    def __init__(self):
        pass

    def get_action(self, state):
        mask = state.legal_actions_mask()
        legals = []
        for i, m in enumerate(mask):
            if m == 1:
                legals.append(i)
        a = np.random.choice(legals)
        return a
