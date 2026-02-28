import argparse
import datetime
import logging

import pyspiel

import agent as agent_module
import game as game_module


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    [lg.removeHandler(h) for h in lg.handlers]
    lg.addHandler(logging.StreamHandler())
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="run_dir", default="runs/test")
    args = parser.parse_args()

    game = pyspiel.load_game("kuhn_poker")
    # game = pyspiel.load_game("tic_tac_toe")
    # game = pyspiel.load_game("phantom_ttt")
    # game = pyspiel.load_game("dark_hex(board_size=5)")
    # game = pyspiel.load_game("dark_chess")
    game = game_module.Game(game)

    import deepcfr
    import escher
    agent_mod = deepcfr
    cfg = agent_mod.Config()
    agent = agent_mod.Agent(game, cfg)

    train_cfg = agent_mod.TrainConfig()
    train_cfg.run_dir = args.run_dir
    train_cfg.device_name = "cpu"
    train_cfg.game = game
    train_cfg.test_agent = agent_module.Random()
    train_cfg.setup()
    logging.info("run_dir \"%s\"", train_cfg.run_dir)

    agent_mod.train(train_cfg, agent)


if __name__ == "__main__":
    main()
