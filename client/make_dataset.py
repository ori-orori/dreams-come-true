import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import tools
import envs.wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd
from client.client_dreamer import Client_Dreamer
from dreamer import Dreamer


def main(config):
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)
    config.norm = getattr(torch.nn, config.norm)
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:

        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
    # train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.envs)]
    

    acts = eval_envs[0].action_space
    config.num_actions = acts.shape[0]



    print("Simulate agent.")
    agent1 = Client_Dreamer(config, logger).to(config.device)
    agent2 = Client_Dreamer(config, logger).to(config.device)
    agent1.requires_grad_(requires_grad=False)
    agent2.requires_grad_(requires_grad=False)
    if (logdir / "latest_model.pt").exists():
        agent1.load_state_dict(torch.load(logdir / "latest_model.pt"))
        agent2.load_state_dict(torch.load(logdir / "latest_model.pt"))
        
        agent1._should_pretrain._once = False
        agent2._should_pretrain._once = False
    state = None
    while agent1._step < config.steps:
        logger.write()
        print("Start evaluation.")
        eval_policy1 = functools.partial(agent1, training=False)
        eval_policy2 = functools.partial(agent2, training=False)
        tools.simulate(eval_policy1, eval_policy2, eval_envs, episodes=config.eval_episode_num)
        # video_pred = agent._wm.video_pred(next(eval_dataset))
        # logger.video("eval_openl", to_np(video_pred))

    # for env in train_envs + eval_envs:
    #     try:
    #         env.close()
    #     except Exception:
    #         pass

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, consfig.batch_size)
    return dataset


def make_env(config, logger, mode, train_eps, eval_eps):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task, mode if "train" in mode else "test", config.action_repeat
        )
        env = wrappers.OneHotAction(env)

    
    elif suite == "soccer":
        import envs.soccer_env as soccer

        env = soccer()

    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    if (mode == "train") or (mode == "eval"):
        callbacks = [
            functools.partial(
                ProcessEpisodeWrap.process_episode,
                config,
                logger,
                mode,
                train_eps,
                eval_eps,
            )
        ]
        env = wrappers.CollectDataset(env, mode, train_eps, callbacks=callbacks)
    env = wrappers.RewardObs(env)
    return env


class ProcessEpisodeWrap:
    eval_scores = []
    eval_lengths = []
    last_step_at_eval = -1
    eval_done = False

    @classmethod
    def process_episode(cls, config, logger, mode, train_eps, eval_eps, episode):
        directory = dict(train=config.traindir, eval=config.evaldir)[mode]
        cache = dict(train=train_eps, eval=eval_eps)[mode]
        # this saved episodes is given as train_eps or eval_eps from next call
        filename = tools.save_episodes(directory, [episode])[0]
        length = len(episode["reward"]) - 1
        score = float(episode["reward"].astype(np.float64).sum())
        video = episode["image"]
        cache[str(filename)] = episode
        if mode == "train":
            total = 0
            for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
                if not config.dataset_size or total <= config.dataset_size - length:
                    total += len(ep["reward"]) - 1
                else:
                    del cache[key]
            logger.scalar("dataset_size", total)
            # use dataset_size as log step for a condition of envs > 1
            log_step = total * config.action_repeat
        elif mode == "eval":
            # keep only last item for saving memory
            while len(cache) > 1:
                # FIFO
                cache.popitem()
            # start counting scores for evaluation
            if cls.last_step_at_eval != logger.step:
                cls.eval_scores = []
                cls.eval_lengths = []
                cls.eval_done = False
                cls.last_step_at_eval = logger.step

            cls.eval_scores.append(score)
            cls.eval_lengths.append(length)
            # ignore if number of eval episodes exceeds eval_episode_num
            if len(cls.eval_scores) < config.eval_episode_num or cls.eval_done:
                return
            score = sum(cls.eval_scores) / len(cls.eval_scores)
            length = sum(cls.eval_lengths) / len(cls.eval_lengths)
            episode_num = len(cls.eval_scores)
            log_step = logger.step
            logger.video(f"{mode}_policy", video[None])
            cls.eval_done = True

        print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        logger.scalar(
            f"{mode}_episodes", len(cache) if mode == "train" else episode_num
        )
        logger.write(step=log_step)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
