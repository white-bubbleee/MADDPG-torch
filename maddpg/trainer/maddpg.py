# -*- coding: utf-8 -*-
# @Date       : 2024/6/30 19:13
# @Auther     : Wang.zr
# @File name  : maddpg.py
# @Description:
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from base.replaybuffer import ReplayBuffer
from base.trainer import Agent, Trainer
import numpy as np
import gym
from ..common.distribution import gen_action_for_discrete, gen_action_for_continuous
from utils.logger import set_logger

logger = set_logger(__name__, output_file="maddpg.log")

# 在任何使用到的地方
# logger.info("Start training")

DATA_TYPE = th.float64


class MADDPGAgent(Agent):
    def __init__(self, name, action_dim, obs_dim, agent_index, args, local_q_func=False, device='cpu'):
        super().__init__(name, action_dim, obs_dim, agent_index, args)
        self.name = name + "_agent_" + str(agent_index)
        self.act_dim = action_dim[agent_index]
        self.obs_dim = obs_dim[agent_index][0]

        self.act_total = sum(action_dim)
        self.obs_total = sum([obs_dim[i][0] for i in range(len(obs_dim))])

        self.device = device

        self.num_units = args.num_units
        self.local_q_func = local_q_func
        self.nums_agents = len(action_dim)
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()

        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=args.lr)

    def build_actor(self, action_bound=None):
        actor = nn.Sequential(
            nn.Linear(self.obs_dim, self.num_units),
            nn.ReLU(),
            nn.Linear(self.num_units, self.num_units),
            nn.ReLU(),
            nn.Linear(self.num_units, self.act_dim),
            # nn.Tanh()
        )

        return actor.to(self.device, dtype=DATA_TYPE)

    def build_critic(self):
        nums_agents = self.nums_agents
        local_q_func = self.local_q_func
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        num_units = self.num_units
        device = self.device

        class Critic(nn.Module):
            def __init__(self):
                super(Critic, self).__init__()
                self.nums_agents = nums_agents
                self.local_q_func = local_q_func
                self.obs_dim = obs_dim
                self.act_dim = act_dim
                self.num_units = num_units
                self.device = device

                if self.local_q_func:  # ddpg
                    self.fc1 = nn.Linear(self.obs_dim + self.act_dim, self.num_units).to(self.device, dtype=DATA_TYPE)
                else:  # maddpg
                    self.fc1 = nn.Linear((self.obs_dim + self.act_dim) * self.nums_agents, self.num_units).to(
                        self.device, dtype=DATA_TYPE)
                self.fc2 = nn.Linear(self.num_units, self.num_units).to(self.device, dtype=DATA_TYPE)
                self.fc3 = nn.Linear(self.num_units, 1).to(self.device, dtype=DATA_TYPE)

            def forward(self, obs, act):
                if self.local_q_func:
                    x = th.cat([obs, act], dim=1)
                else:
                    obs = np.concatenate(obs, axis=1)
                    obs_all = th.from_numpy(obs).to(self.device, dtype=DATA_TYPE)
                    if not isinstance(act[0], th.Tensor):
                        act = np.concatenate(act, axis=1)
                        act_all = th.from_numpy(act).to(self.device, dtype=DATA_TYPE)
                    else:
                        act_all = th.cat(act, dim=1).to(self.device, dtype=DATA_TYPE)
                    x = th.cat([obs_all, act_all], dim=1).to(self.device, dtype=DATA_TYPE)
                x = th.relu(self.fc1(x))
                x = th.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        critic = Critic()

        return critic

    def agent_action(self, obs):
        return self.actor(obs)

    def agent_critic(self, obs, act):
        return self.critic(obs, act)

    def agent_target_action(self, obs):
        return self.target_actor(obs)

    def agent_target_critic(self, obs, act):
        return self.target_critic(obs, act)

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


class MADDPGTrainer(Trainer):
    def __init__(self, name, obs_dims, action_space, agent_index, args, local_q_func=False):
        super().__init__(name)
        self.name = name
        self.args = args
        self.agent_index = agent_index
        self.nums = len(obs_dims)

        self.device = "cuda:0" if args.use_gpu and th.cuda.is_available() else "cpu"

        # ======================= env_config preprocess =========================
        self.action_space = action_space
        if isinstance(action_space[0], gym.spaces.Box):
            self.act_dims = [self.action_space[i].shape[0] for i in range(self.nums)]
            self.action_out_func = gen_action_for_continuous
        elif isinstance(action_space[0], gym.spaces.Discrete):
            self.act_dims = [self.action_space[i].n for i in range(self.nums)]
            self.action_out_func = gen_action_for_discrete

        # ====================== hyperparameters =========================
        self.local_q_func = local_q_func
        if self.local_q_func:
            logger.info(f"Init {agent_index} is using DDPG algorithm")
        else:
            logger.info(f"Init {agent_index} is using MADDPG algorithm")
        self.grad_norm_clip = args.grad_norm_clip
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size

        self.agent = MADDPGAgent(name, self.act_dims, obs_dims, agent_index, args, local_q_func=local_q_func,
                                 device=self.device)
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        # ====================initialize target networks====================
        self.update_target(self.agent.target_actor.parameters(), self.agent.actor.parameters(), tau=self.tau)
        self.update_target(self.agent.target_critic.parameters(), self.agent.critic.parameters(), tau=self.tau)

    def train(self, trainers, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return

        if not t % 100 == 0:  # only update every 100 steps
            return

        obs_n, action_n, reward_i, next_obs_n, done_i = self.sample_batch_for_pretrain(trainers)
        # ======================== train critic ==========================

        target_actions = [trainer.get_target_action(next_obs_n[i]) for i, trainer in enumerate(trainers)]
        #  ============= target ===========
        target_q_input = (next_obs_n, target_actions)  # global info
        if self.local_q_func:
            target_q_input = (next_obs_n[self.agent_index], target_actions[self.agent_index])
        target_q = self.agent.agent_target_critic(*target_q_input)
        reward_i = th.from_numpy(reward_i).to(self.device, dtype=DATA_TYPE)
        done_i = th.from_numpy(done_i).to(self.device, dtype=DATA_TYPE)
        y = reward_i + self.gamma * (1 - done_i) * target_q  # target
        # ============= current ===========
        q_input = (obs_n, action_n)  # global info
        if self.local_q_func:  # local info
            q_input = (obs_n[self.agent_index], action_n[self.agent_index])
        q = self.agent.agent_critic(*q_input)
        critic_loss = F.mse_loss(y, q)
        self.agent.critic_optimizer.zero_grad()
        th.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.grad_norm_clip)
        critic_loss.backward()
        self.agent.critic_optimizer.step()

        # ========================= train actor ===========================
        _action_n = []
        for i, trainer in enumerate(trainers):
            _action = trainer.get_action(obs_n[i])
            _action_n.append(_action)
        q_input = (obs_n, _action_n)
        if self.local_q_func:
            q_input = (obs_n[self.agent_index], _action_n[self.agent_index])
        p_reg = th.mean(th.square(_action_n[self.agent_index]))  # regularization
        actor_loss = -th.mean(self.agent.agent_critic(*q_input)) + p_reg * 1e-3

        self.agent.actor_optimizer.zero_grad()
        th.nn.utils.clip_grad_value_(self.agent.critic.parameters(), self.grad_norm_clip)
        actor_loss.backward()
        self.agent.actor_optimizer.step()

        # ======================= update target networks ===================
        self.update_target(self.agent.target_actor.parameters(), self.agent.actor.parameters(), self.tau)
        self.update_target(self.agent.target_critic.parameters(), self.agent.critic.parameters(), self.tau)

    def pretrain(self):
        self.replay_sample_index = None

    def save_model(self, path):
        pass

    def locd_model(self, path):
        self.agent.load_model(path)

    def get_action(self, state):
        # return tf.cond(
        #     tf.rank(state) == 1,
        #     lambda: self.action_out_func(self.agent.agent_action(state.squeeze(axis=0))[0]),
        #     lambda: self.action_out_func(self.agent.agent_action(state))
        # )
        # if state.ndim == 1:
        #     state = np.expand_dims(state, axis=0)
        #     action_re = self.action_out_func(self.agent.actor(state)[0])
        # else:
        #     action_re = self.action_out_func(self.agent.actor(state))
        state = th.tensor(state, dtype=DATA_TYPE).to(self.device)
        return self.action_out_func(self.agent.actor(state))

    def get_target_action(self, state):
        state = th.tensor(state, dtype=DATA_TYPE).to(self.device)
        return self.action_out_func(self.agent.target_actor(state))

    def update_target(self, target_weights, weights, tau):
        for (target, weight) in zip(target_weights, weights):
            target.data.copy_(weight * tau + target * (1 - tau))

    def experience(self, state, action, reward, next_state, done, terminal):
        self.replay_buffer.add(state, action, reward, next_state, float(done))

    def sample_batch_for_pretrain(self, trainers):
        if self.replay_sample_index is None:
            self.replay_sample_index = self.replay_buffer.make_index(self.batch_size)
        obs_n, action_n, next_obs_n = [], [], []
        reward_i, done_i = None, None
        for i, trainer in enumerate(trainers):
            obs, act, rew, next_obs, done = trainer.replay_buffer.sample_index(self.replay_sample_index)

            obs_n.append(obs)
            action_n.append(act)
            next_obs_n.append(next_obs)

            if self.agent_index == i:
                done_i = done
                reward_i = rew
        return obs_n, action_n, reward_i[:, np.newaxis], next_obs_n, done_i[:, np.newaxis]
