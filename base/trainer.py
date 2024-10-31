# -*- coding: utf-8 -*-
# @Date       : 2024/5/23 10:02
# @Auther     : Wang.zr
# @File name  : trainer.py
# @Description:

class Agent:

    def __init__(self, name, action_dim, obs_dim, agent_index, args):
        pass

    def build_actor(self):
        raise NotImplementedError

    def build_critic(self):
        raise NotImplementedError


class NormalAgent:

    def __init__(self, name, action_dim, obs_dim, agent_index, args):
        pass

    def build_policy(self):
        raise NotImplementedError

    def build_critic(self):
        raise NotImplementedError


class Trainer:

    def __init__(self, name):
        self.name = name
        pass

    def update_target(self, target_weights, weights, tau):
        for (target, weight) in zip(target_weights, weights):
            target.data.copy_(weight * tau + target * (1 - tau))
