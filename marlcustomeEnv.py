#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：UARL 
@File    ：marlcustomeEnv.py
@Author  ：Jiansheng LI
@Date    ：2023/12/31 0:52 
'''

from typing import Any
import copy
import gymnasium as gym
import numpy as np
import torch as th
from module.environment import Environment


class MarlCustomEnv(Environment):
    def __init__(self, sce):

        super().__init__(sce)
        # TODO check Encoder
        self.history_channel_information = None
        self.dtype = np.float32
        # self.enc1 = OneHotEncoder(sparse=False)
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))

        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        # self.TxPowerVector = np.array([b.Transmit_Power_dBm() for b in self.BSs])
        # b, k, u = self.BS_num, sce.nRBs, sce.nUEs
        # self.MatrixE = np.zeros((u, k * u))

        # for every basestation, its user candidate are the set of user locating in its signal coverage zone.
        # thus, the action space of every agent is the box with dimension of number of users in its signal coverage zone.
        self.action_spaces = []
        self.observation_spaces = []

        totalActionDimension = 0
        for b in self.BSs:
            totalActionDimension += len(b.UE_set)
            self.action_spaces.append(gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set),)))
            self.observation_spaces.append(
                gym.spaces.box.Box(np.array([-np.inf] * len(b.UE_set)), np.array([np.inf] * len(b.UE_set)),
                                   (len(b.UE_set),),
                                   dtype=self.dtype))
        self.action_space = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension,))

        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(len(self.BSs), len(self.UEs)),
                                                    dtype=self.dtype)
        self.MaxReward = 0
        self.StopThreshold = 0.05
        self.LastReward = 0
        self.LastAction = None

    def step(self, actions):
        return self.MARLstep(actions)

    def reset(self, seed=None, options=None):
        observation, info = self.MARLreset(seed, options)
        return observation, info



class MarlCustomEnv2(MarlCustomEnv):
    def __init__(self, sce, env_instance: MarlCustomEnv = None):

        self.flag = True
        # if we want to use the existed MarlCustomEnv instance info, we need to pass the sce and the env_instance to the super class
        if env_instance is None:
            super().__init__(sce)
        else:
            self.__setstate__(env_instance.__getstate__())
            self.sce = sce

        self.history_channel_information = None
        self.dtype = np.float32
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))
        self.use_action_projection = False
        self.StopThreshold = 0.05
        self.LastAction = None
        # pre calculate the distance matrix
        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))
        # define action space and observation space of two kinds of actions
        self.user_association_action_spaces = []
        self.rbg_assignment_action_spaces = []
        self.user_association_observation_spaces = []
        self.rbg_assignment_observation_spaces = []
        totalActionDimension = 0

        for b in self.BSs:
            # for actor policy input parameter
            totalActionDimension += len(b.UE_set) + len(b.UE_set) * self.sce.nRBs
            # totalActionDimension_ua += len(b.UE_set)
            self.user_association_action_spaces.append(gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set),)))
            self.user_association_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf, (len(b.UE_set),), dtype=self.dtype))

            # totalActionDimension_rbg += len(b.UE_set) * self.sce.nRBs
            self.rbg_assignment_action_spaces.append(
                gym.spaces.box.Box(low=0, high=1, shape=(len(b.UE_set) * self.sce.nRBs,)))
            self.rbg_assignment_observation_spaces.append(
                gym.spaces.box.Box(-np.inf, np.inf,
                                   (len(b.UE_set) * self.sce.nRBs,),
                                   dtype=self.dtype)
            )
        
        self.action_space = gym.spaces.box.Box(low=0, high=1, shape=(totalActionDimension,))
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf,
                                                    shape=(len(self.BSs) * len(self.UEs) * self.sce.nRBs,),
                                                    dtype=self.dtype)



    def distribute_actions(self, actions, mode=0) -> list:
        """
        :param actions: actions_total
        :return: [[agent1 ua action, agent2 ua action, ...] [agent1 rbg action, agent 2 rbg action, ...]]
        """
        actions_user_association_RBG_assignment = actions
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        return [bs_wise_user_association_action, bs_wise_RBG_assignment_action]

    def MARLstep(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[Any, float, bool, bool, dict[Any, Any]]:
        return self.MARLstep_withCurrentH(actions_user_association_RBG_assignment)

    def project_rbg_action(self, bs_wise_actions, threshold=0.5):
        if self.sce.rbg_N_b is None:
            self.sce.rbg_N_b = 3
        N_rb = self.sce.rbg_N_b
        action_rbg_bs_wise_proj = []
        for tensor in bs_wise_actions:
            if not isinstance(tensor, th.Tensor):
                tensor = th.tensor(tensor)
            if threshold is None:
                _, indices = th.topk(tensor, N_rb, dim=-1)
                new_tensor = th.zeros_like(tensor)
                new_tensor.scatter_(-1, indices, 1)
                action_rbg_bs_wise_proj.append(new_tensor)
            else:
                new_tensor = th.zeros_like(tensor)
                _, topk_indices = th.topk(tensor, N_rb, dim=-1)
                new_tensor.scatter_(-1, topk_indices, 1)
                new_tensor[tensor < threshold] = 0
                action_rbg_bs_wise_proj.append(new_tensor)
        return action_rbg_bs_wise_proj

    def MARLstep_withCurrentH(self, actions_user_association_RBG_assignment: np.ndarray) -> tuple[
        Any, float, bool, bool, dict[Any, Any]]:
        """
        MARL step function
        :param actions_user_association_RBG_assignment:
        list[list[user_association shape(env_idx=1, len(b.ue_set)],
            list[RBG_assignment shape(env_idx=1, sum(len(b.ue_set))]]
        :return:
        - observation_total: shape(len(self.BSs)*len(self.UEs);
        - reward: one reward for all agents;
        - terminated: one terminated for all agents
        """

        # rbg_assignment are a bunch of single_bs_rbg_assignment, thus we need basestation-wisely separate them.
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))

        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        if self.use_action_projection:
            bs_wise_RBG_assignment_action = self.project_rbg_action(bs_wise_RBG_assignment_action)

        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        # [np.array() for i in range(self.sce.nRBs)]  # store all users' the signal power in every sub-carrier
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        # channal_power_set = [[[0 for _ in range(self.sce.nRBs)] for _ in range(self.sce.nUEs)] for _ in range(
        # self.sce.nBSs)]
        # a_b_k_u = 1 / self.sce.nRBs  # a 平均化处理
        assert self.history_channel_information is not None
        # if self.history_channel_information is not None:
        H_dB = self.history_channel_information.reshape((self.BS_num, self.sce.nUEs, self.sce.nRBs), )
        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index].squeeze()
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]

                    _, channel_power_dBm = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    # s_b_u *
                    signal_power_set[rb_index][global_u_index] += s_b_u * a_b_k_u * b.Transmit_Power() / (
                            10 ** (H_dB[b_index, global_u_index, rb_index] / 10))
                    # print(f'sbku before:{signal_power_set[rb_index][global_u_index]}\nafter{signal_power_set[rb_index][global_u_index]*s_b_u}')
                    # H[b_index,global_u_index,rb_index] * b.Transmit_Power_dBm()
                    # 注意 H_dB 是 fading - pathloss
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power_dBm
        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        # interference_sum = interference_sum.reshape(-1,1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)

        interference_m = interference_sum_m - signal_power_set + Noise
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(1, -1)
        reward = total_rate
        self.history_channel_information = obs

        terminated, truncated, info = False, False, {}

        return obs, reward, terminated, truncated, info


    def get_init_obs(self, actions_user_association_RBG_assignment: np.ndarray):
        stride0 = int(self.action_space.shape[0] / (self.sce.nRBs + 1))
        bs_wise_user_association_action = []
        actions_user_association_total = actions_user_association_RBG_assignment[:stride0]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set)
            bs_wise_user_association_action.append(
                actions_user_association_total[..., index:index + stride])
            index += stride

        bs_wise_RBG_assignment_action = []
        actions_RBG_assignment_total = actions_user_association_RBG_assignment[stride0:]
        index = 0
        for i, b in enumerate(self.BSs):
            stride = len(b.UE_set) * self.sce.nRBs
            bs_wise_RBG_assignment_action.append(
                actions_RBG_assignment_total[..., index:index + stride].reshape(len(b.UE_set), self.sce.nRBs))
            index += stride

        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))

        for b_index, b in enumerate(self.BSs):
            s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            a_assignment_bs = bs_wise_RBG_assignment_action[b_index]
            for local_u_index, (s_b_u, s_a_b_k_u) in enumerate(zip(s_assignment_bs, a_assignment_bs)):
                global_u_index = b.UE_set[local_u_index] - 1
                # s_a_b_k_u = s_a_b_k_u.squeeze()
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = s_a_b_k_u[rb_index]
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    # = self.test_cal_Receive_Power_without_fading(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power
        obs = channal_power_set.reshape(1, -1)
        return obs

    def MARLreset(self, seed=None, options=None):

        ua_act = []
        for i in self.user_association_observation_spaces:
            ua_act.append(i.sample())
        ua_act = np.concatenate(ua_act)

        rbg_act = []
        for i in self.rbg_assignment_observation_spaces:
            rbg_act.append(i.sample())
        rbg_act = np.concatenate(rbg_act)
        act = np.concatenate([ua_act, rbg_act])

        obs = self.get_init_obs(act)
        obs = np.squeeze(obs)
        self.history_channel_information = obs

        observation, info = np.array(obs), {}
        return observation, info



class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
                self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        # self edit
        observation, reward, done, truncated, info = self.env.step(action)
        # original
        # observation, reward, terminated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
            # self edit
            truncated = True
        # self edit
        return observation, reward, done, truncated, info
        # orignal
        # return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


