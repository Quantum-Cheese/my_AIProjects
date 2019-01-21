from collections import namedtuple
from collections import deque
import numpy as np

Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])

"""存放经验元组的缓存区"""
class ReplayBuffer:

    def __init__(self, size=1000):
        """初始化Buffer"""
        self.size = size  # maximum size of buffer
        self.memory = deque(maxlen=self.size)  # internal memory (list)
        self.probabilities=[]  # 各经验元组的取样概率

    def add(self, state, action, reward, next_state, done):
        """添加一组新的经验"""

        # 从左侧添加，如果空间已满，挤出右侧末端元组
        ex=Experience(state=state,action=action,reward=reward,next_state=next_state,done=done)
        self.memory.appendleft(ex)

        # 根据reward大小降序排列
        dq=sorted(self.memory,key=lambda x:x[2],reverse=True)
        self.memory=deque(dq,maxlen=self.size)

        # 更新各元组的取样概率
        self.set_probabilities()

    def sample(self, batch_size=64):
        """根据概率probabilities进行取样"""

        sample_indexs=[] # 取样的元组索引
        full_indexs=np.arange(self.__len__()) # memory中现有的所有元组索引
        # 按照probabilities中的取样概率确定取样的元组索引
        for i in range(batch_size):
            ind = np.random.choice(full_indexs, p=self.probabilities)
            sample_indexs.append(ind)

        return [self.memory[i] for i in sample_indexs]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def set_probabilities(self):
        # 根据优先级(排序后的memory中元组的顺序)设定新元组取样概率
        length=self.__len__()
        bond_1 = length // 4
        bond_2 = length // 2

        self.probabilities = [0 for i in range(0, length)]
        for i in range(0, bond_1):
            self.probabilities[i] = 0.7 / bond_1
        for i in range(bond_1, bond_2):
            self.probabilities[i] = 0.2 / (bond_2 - bond_1)
        for i in range(bond_2, length):
            self.probabilities[i] = 0.1 / (length - bond_2)

