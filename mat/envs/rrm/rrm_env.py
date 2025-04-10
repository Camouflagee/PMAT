from mat.envs.football.multiagentenv import MultiAgentEnv
import numpy as np
import torch

class RRMEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 从环境参数中获取配置
        args = kwargs["env_args"]
        self.n_bs = args.get("n_bs", 3)  # 基站数量
        self.n_ue = args.get("n_ue", 10)  # 用户数量 
        self.n_rb = args.get("n_rb", 4)   # 资源块数量
        self.n_agents = self.n_bs         # 每个基站作为一个智能体
        self.episode_limit = args.get("episode_limit", 100)
        
        # 定义动作空间和观察空间
        self.n_actions = self.n_ue * self.n_rb  # 每个基站可以选择为哪个用户分配哪个资源块
        
        # 观察空间包括:
        # - 信道状态信息 (n_ue * n_rb)
        # - 用户关联状态 (n_ue)
        # - 资源块使用状态 (n_rb) 
        self.obs_dim = self.n_ue * self.n_rb + self.n_ue + self.n_rb
        
        # 初始化信道状态等环境信息
        self.reset()
        
    def reset(self):
        """重置环境状态"""
        self.step_count = 0
        
        # 随机初始化信道状态
        self.channel_states = np.random.randn(self.n_bs, self.n_ue, self.n_rb)
        
        # 初始化用户关联和资源块分配
        self.user_association = np.zeros((self.n_bs, self.n_ue))
        self.rb_allocation = np.zeros((self.n_bs, self.n_ue, self.n_rb))
        
        # 获取初始观察
        obs = self.get_obs()
        state = self.get_state()
        
        return obs, state
        
    def step(self, actions):
        """环境交互一步"""
        self.step_count += 1
        
        # 解析智能体的动作
        self._handle_actions(actions)
        
        # 计算奖励(总和速率)
        reward = self._compute_reward()
        
        # 判断是否结束
        done = (self.step_count >= self.episode_limit)
        
        # 获取新的观察和状态
        obs = self.get_obs()
        state = self.get_state()
        
        return reward, done, {'state': state}
        
    def get_obs(self):
        """返回每个智能体的观察"""
        obs = []
        for agent_id in range(self.n_agents):
            agent_obs = np.concatenate([
                self.channel_states[agent_id].flatten(),
                self.user_association[agent_id],
                np.sum(self.rb_allocation[agent_id], axis=0)
            ])
            obs.append(agent_obs)
        return obs
        
    def get_state(self):
        """返回全局状态"""
        return np.concatenate([
            self.channel_states.flatten(),
            self.user_association.flatten(),
            self.rb_allocation.flatten()
        ])
        
    def _handle_actions(self, actions):
        """处理智能体动作"""
        for agent_id, action in enumerate(actions):
            # 将动作解析为用户选择和资源块分配
            user_idx = action // self.n_rb
            rb_idx = action % self.n_rb
            
            # 更新用户关联和资源块分配
            self.user_association[agent_id] = 0
            self.user_association[agent_id, user_idx] = 1
            
            self.rb_allocation[agent_id] = 0
            self.rb_allocation[agent_id, user_idx, rb_idx] = 1
            
    def _compute_reward(self):
        """计算系统总和速率作为奖励"""
        total_rate = 0
        noise = 1e-10
        
        for u in range(self.n_ue):
            for b in range(self.n_bs):
                if self.user_association[b,u]:
                    for k in range(self.n_rb):
                        if self.rb_allocation[b,u,k]:
                            # 计算信噪比
                            signal = np.abs(self.channel_states[b,u,k])**2
                            interference = 0
                            for b2 in range(self.n_bs):
                                if b2 != b:
                                    interference += np.abs(self.channel_states[b2,u,k])**2 * self.rb_allocation[b2,:,k].sum()
                            
                            sinr = signal / (interference + noise)
                            rate = np.log2(1 + sinr)
                            total_rate += rate
                            
        return total_rate