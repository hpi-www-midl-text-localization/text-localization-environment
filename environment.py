import gym
from PIL import ImageDraw


class TextLocEnv(gym.Env):

    def __init__(self, image, true_bboxes):
        self.alpha = 0.2

        self.image = image
        self.true_bboxes = true_bboxes
        self.history = []
        self.bbox = (0, 0, image.width, image.height)
        self.state = self.compute_state()

    def step(self, action):
        """Execute an action and return
            state - the next state,
            reward - the reward,
            done - whether a terminal state was reached,
            info - any additional info"""

        return 0, 0, True, {}

    def reset(self):
        """Reset the environment to its initial state (the bounding box covers the entire image"""
        self.history = []
        self.bbox = (0, 0, self.image.width, self.image.height)
        self.state = self.compute_state()

        return self.state

    def render(self, mode='human'):
        """Render the current state"""
        copy = self.image.copy()
        draw = ImageDraw.Draw(copy)
        draw.rectangle(self.bbox, outline=(255, 255, 255))
        copy.show()

    def compute_state(self):
        """Compute the state from the image, the bounding box and the action history"""
        return 0
