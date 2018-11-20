import gym


class TextLocEnv(gym.Env):

    def __init__(self, image):
        pass

    def step(self, action):
        """Execute an action and return
            state - the next state,
            reward - the reward,
            done - whether a terminal state was reached,
            info - any additional info"""
        return 0, 0, True, {}

    def reset(self):
        """Reset the environment to its initial state"""
        pass

    def render(self, mode='human'):
        """Render the current state"""
        pass
