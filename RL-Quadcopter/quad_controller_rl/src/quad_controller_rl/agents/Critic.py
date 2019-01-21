from keras import layers, models, optimizers,regularizers,initializers
from keras import backend as K
import numpy as np

"""评论者模型"""
class Critic:

    def __init__(self, state_size, action_size):

        """
       参数
        ======
            state_size (int): 每个state的维度
            action_size (int): 每个action的维度
        """
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        """构建评论者网络：把 (state, action) 对映射成 Q-value."""
        # 定义输入层
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        reg=regularizers.l2(0.00001)
        
        # state子网络的隐藏层
        net_states = layers.Dense(units=16, activation='relu',kernel_regularizer=reg)(states)
        net_states = layers.Dense(units=16, activation='relu',kernel_regularizer=reg)(net_states)

        # action自网络的隐藏层
        net_actions = layers.Dense(units=16, activation='relu',kernel_regularizer=reg)(actions)
        net_actions = layers.Dense(units=16, activation='relu',kernel_regularizer=reg)(net_actions)

        # 合并state 和 action 子网络
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # 输出层，输出 Q 值
        Q_values = layers.Dense(units=1,kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),name='q_values')(net)

        # 构建 Keras 模型
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # 定义优化器和损失函数，编译模型
        optimizer = optimizers.Adadelta()
        self.model.compile(optimizer=optimizer, loss='mse')

        # 计算action的梯度 (Q_value 对 action求导)
        action_gradients = K.gradients(Q_values, actions)

        # 定义一个专门获取action梯度的函数 (以供actor模型调用)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)




