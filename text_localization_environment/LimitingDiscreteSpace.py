import numpy as np
from gym import spaces


class LimitingDiscrete(spaces.Discrete):
    def __init__(self, n):
        super().__init__(n)
        self.available_actions = range(0, n)

    def sample(self):
        return spaces.np_random.choice(self.available_actions)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int in self.available_actions

    def set_available_actions(self, available_actions):
        self.available_actions = available_actions
        return self.available_actions

    def reset_available_actions(self):
        self.available_actions = range(0, self.n)
        return self.available_actions

    def enable_actions(self, actions):
        """ You would call this method inside your environment to enable actions"""
        self.available_actions = self.available_actions.append(actions)
        return self.available_actions

    def __repr__(self):
        return "LimitedDiscrete(%s of %d)" % (', '.join(self.available_actions), self.n)

    def __eq__(self, other):
        return self.n == other.n and self.available_actions == other.available_actions
