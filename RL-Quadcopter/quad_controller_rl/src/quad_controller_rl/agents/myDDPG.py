from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.Actor import Actor
from quad_controller_rl.agents.Critic import Critic
from quad_controller_rl.agents.ReplayBuffer import ReplayBuffer
from quad_controller_rl.agents.OUNoise import OUNoise

import numpy as np
import matplotlib.pyplot as plt


class myDDPG(BaseAgent):

    def __init__(self, task):

        self.task = task
        self.state_size = np.prod(self.task.observation_space.shape)
        self.action_size = np.prod(self.task.action_space.shape)

        # 限制状态和动作空间
        self.limit_state_size = 3
        self.limit_action_size = 1
        self.action_low = self.task.action_space.low[2]
        self.action_high = self.task.action_space.high[2]

        # 设定上个动作和状态的初始值:a(t-1)和s(t-1)
        self.last_state = None
        self.last_action = None

        # 设定Actor和Critic的local_model和target_model
        self.local_actor = Actor(self.limit_state_size, self.limit_action_size, self.action_low, self.action_high)
        self.local_critic = Critic(self.limit_state_size, self.limit_action_size)
        self.target_actor = Actor(self.limit_state_size, self.limit_action_size, self.action_low, self.action_high)
        self.target_critic = Critic(self.limit_state_size, self.limit_action_size)
        # 设定超参数
        self.batch_size = 150
        self.buffer_size = 10000
        self.soft_params = 0.005
        self.gamma = 0.99
        # 设定缓存区
        self.memory=ReplayBuffer(self.buffer_size)
        # 设定随机探索噪点
        self.noise = OUNoise(self.limit_action_size)

    """ 接收task传来的上一个动作a(t-1)引发的R(t)和S(t)，选择当前动作a(t)，并通过学习优化策略"""
    def step(self, state, reward, done):

        state = self.preprocess_state(state)

        # 根据当前的状态s(t)选择动作a(t)
        action = self.act(state)

        # 把经验元组存储到缓冲区
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
        self.last_state = state
        self.last_action = action

        # 当 memory 中有足够的经验，从中批量取样进行学习
        memorySize = self.memory.__len__()
        if memorySize >= self.batch_size:
            expriences = self.memory.sample(batch_size=self.batch_size)
            self.learn(expriences)
            
        # 最后返回完整的动作向量
        complete_action = self.postprocess_action(action)
        return complete_action

    def act(self, state):
        # 调用Actor,把当前状态（向量）作为输入，得到根据当前策略所选择的动作（向量）
        input_state = np.array(state)
        # 加入随机噪点
        action = self.local_actor.model.predict(input_state) + self.noise.sample()  
        return action.astype(np.float32)

    def learn(self, expriences):
        # 对入参expriences进行处理，拆分出states,actions,rewards,dones,next_states
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for ex in expriences:
            if ex is not None:
                states.append(ex.state)
                actions.append(ex.action)
                rewards.append(ex.reward)
                next_states.append(ex.next_state)
                dones.append(ex.done)
        # 转换数据格式
        states, actions, rewards, dones, next_states = np.array(states).astype(np.float32), np.array(actions).astype(
            np.float32) \
            , np.array(rewards).astype(np.float32), np.array(dones).astype(np.uint8), np.array(next_states).astype(
            np.float32)

        """训练 Critic，更新模型参数 (用target_model作为标签训练local_model)"""

        # 用批量 s(t+1) 输入target_actor预测得到批量 a(t+1)
        input_next_states = np.reshape(next_states, (len(next_states), self.limit_state_size))
        next_actions = self.target_actor.model.predict(input_next_states).astype(np.float32)

        # 批量s(t+1)和a(t+1)作为输入，通过target_critic 得到Q(s(t+1),a(t+1))
        input_next_actions = np.reshape(next_actions, (len(next_actions), self.limit_action_size)).astype(np.float32)
        Q_sa = self.target_critic.model.predict([input_next_states, input_next_actions]).astype(np.float32)
        # 计算得到Q_targets
        Q_targets = Q_sa * self.gamma * (1 - np.reshape(dones, (-1, 1)).astype(np.uint8)) + np.reshape(rewards, (
        len(rewards), 1)).astype(np.float32)

        # 用Q_targets做标签，训练local_critic模型参数
        input_states = np.reshape(states, (len(states), self.limit_state_size))
        input_actions = np.reshape(actions, (len(actions), self.limit_action_size)).astype(np.float32)
        self.local_critic.model.fit([input_states, input_actions], Q_targets)

        """ 训练 Actor，更新模型参数"""

        # 由 local_critic 得到策略参数梯度 action_gradients
        g_a = self.local_critic.get_action_gradients(inputs=[input_states,input_actions,0])
        action_gradient=np.reshape(g_a,(-1,self.limit_action_size))
        # 用action_gradients训练local_actor模型
        self.local_actor.train_fn(inputs=[input_states, action_gradient, 1])

        """ soft_update """
        self.soft_update(self.local_critic, self.target_critic)
        self.soft_update(self.local_actor, self.target_actor)

    """限制状态空间，把task传来的8维状态向量降到3维（z,vel,linear_acceleration.z）"""
    def preprocess_state(self, raw_state):
        state = np.array([raw_state])
        return state[:, 2:5]

    """把1维动作向量扩展到6维（剩下维度都补零），以供返回给task"""

    def postprocess_action(self, action):
        complete_action = np.zeros((1, self.action_size))  # shape: (6,)
        complete_action[:,2] = action
        return complete_action[0]

    """软更新，用local模型的权重更新target模型权重"""

    def soft_update(self, local, target):
        local_weights = np.array(local.model.get_weights())
        target_weights = np.array(target.model.get_weights())

        new_weights = self.soft_params * local_weights + (1 - self.soft_params) * target_weights
        target.model.set_weights(new_weights)

  



