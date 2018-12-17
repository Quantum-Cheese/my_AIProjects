import random
from collections import namedtuple

Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])

"""存放经验元组的缓存区"""
class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Note: If memory is full, start overwriting from the beginning

        if len(self.memory)>=self.size:
            self.memory=[]
        ex=Experience(state=state,action=action,reward=reward,next_state=next_state,done=done)

        self.memory.append(ex)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        indexs=random.sample(range(0,len(self.memory)),batch_size)
        return [self.memory[i] for i in indexs]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



