import argparse
import datetime
import logging

import pyspiel

import agent as agent_module
import game as game_module
import escher


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

    cfg = escher.Config()
    # cfg.value_net = [256, 256]
    # cfg.value_batch_size = 2048
    # cfg.value_batch_steps = 5000
    # cfg.regret_net = [256, 256]
    # cfg.regret_batch_size = 2048
    # cfg.regret_batch_steps = 5000
    # cfg.avg_policy_net = [256, 256]
    # cfg.avg_policy_batch_size = 2048
    # cfg.avg_policy_batch_steps = 10000
    agent = escher.Agent(game, cfg)

    train_cfg = escher.TrainConfig()
    train_cfg.run_dir = args.run_dir
    train_cfg.device_name = "cpu"
    train_cfg.game = game
    train_cfg.test_agent = agent_module.Random()
    # train_cfg.test_every = 5
    train_cfg.nashconv = True
    train_cfg.setup()
    logging.info("run_dir \"%s\"", train_cfg.run_dir)

    escher.train(train_cfg, agent)


if __name__ == "__main__":
    main()
