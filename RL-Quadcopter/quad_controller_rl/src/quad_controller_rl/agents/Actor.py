from keras import layers, models, optimizers,regularizers,initializers
from keras import backend as K
import numpy as np

"""行动者模型"""
class Actor:

    def __init__(self, state_size, action_size, action_low, action_high):

        """参数
        ======
            state_size (int): 每个state的维度
            action_size (int): 每个action的维度
            action_low (array): action中每个维度的最小值
            action_high (array): action中每个维度的最大值
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low


        self.build_model()

    def build_model(self):
        """构建actor网络模型"""
        # 定义输入层 (states)
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # 批归一化
        net = layers.BatchNormalization()(states)
        
        # 隐藏层
        net = layers.Dense(units=16, activation='relu')(net)
        net = layers.Dense(units=16, activation='relu')(net)
        net = layers.Dense(units=16, activation='tanh')(net)

        # 输出层
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),name='raw_actions')(net)

        # 把原始输出（0到1之间）值转化为真实值
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # 构建keras模型
        self.model = models.Model(inputs=states, outputs=actions)

        # 使用action的梯度定义损失函数
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # 定义优化器，训练函数
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)



