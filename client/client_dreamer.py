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

import exploration as expl
import models
import tools
# import envs.wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Client_Dreamer(nn.Module):
    def __init__(self, config):
        super(Client_Dreamer, self).__init__()
        self._config = config
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        logdir = pathlib.Path(config.logdir).expanduser()
        self._step = count_steps(logdir / "train_eps")
        if isinstance(config.act, str):
            config.act = getattr(torch.nn, config.act)
            config.norm = getattr(torch.nn, config.norm)
        self._update_count = 0
        # Schedules.
        # config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(
        #     x, self._step
        # )
        # config.actor_state_entropy = (
        #     lambda x=config.actor_state_entropy: tools.schedule(x, self._step)
        # )
        # config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(
        #     x, self._step
        # )
        self._wm = models.WorldModel(self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if config.compile:
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)


    def __call__(self, obs, reset, state=None, reward=None, training=False):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]

        policy_output, state = self._policy(obs, state, training)

        print(policy_output)

        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)

        actor = self._task_behavior.actor(feat)
        action = actor.mode()
        # elif self._should_expl(self._step):
        #     actor = self._expl_behavior.actor(feat)
        #     action = actor.sample()
        # else:
        #     actor = self._task_behavior.actor(feat)
        #     action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )


        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)

        policy = np.argmax(policy_output["action"])

        return policy, state

    def _exploration(self, action, training):
        amount = self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
        raise NotImplementedError(self._config.action_noise)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))