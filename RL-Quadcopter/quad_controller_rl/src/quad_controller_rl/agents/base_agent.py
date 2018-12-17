"""Generic base class for reinforcement learning agents."""

class BaseAgent:
    """Generic base class for reinforcement reinforcement agents."""

    def __init__(self, task):
        """Initialize policy and other agent parameters.

        Should be able to access the following (OpenAI Gym spaces):
            task.observation_space  # i.e. state space
            task.action_space
        """
        self.task = task
        self.state_size = np.prod(self.task.observation_space.shape)
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = np.prod(self.task.action_space.shape)
        self.action_low=self.task.action_space.low
        self.action_high=self.task.action_space.high

    def step(self, state, reward, done):
        """Process state, reward, done flag, and return an action.

        Params
        ======
        - state: current state vector as NumPy array, compatible with task's state space
        - reward: last reward received
        - done: whether this episode is complete

        Returns
        =======
        - action: desired action vector as NumPy array, compatible with task's action space
        """
        raise NotImplementedError("{} must override step()".format(self.__class__.__name__))
