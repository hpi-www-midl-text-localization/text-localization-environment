import gym
from gym import spaces
from chainer.links import VGG16Layers
from PIL import ImageDraw
from PIL.Image import LANCZOS
import numpy as np


class TextLocEnv(gym.Env):

    HISTORY_LENGTH = 10

    def __init__(self, image, true_bboxes):
        self.alpha = 0.2
        self.feature_extractor = VGG16Layers()
        self.action_space = spaces.Discrete(9)
        self.action_set = {0: self.up,
                           1: self.down,
                           2: self.left,
                           3: self.right,
                           4: self.zoom_in,
                           5: self.zoom_out,
                           6: self.wider,
                           7: self.taller,
                           8: self.trigger
                           }

        # self.feature_extractor.to_gpu(0)

        self.image = image
        self.true_bboxes = true_bboxes
        self.history = []
        self.bbox = np.array([0, 0, image.width, image.height])
        self.state = self.compute_state()
        self.done = False

    def step(self, action):
        """Execute an action and return
            state - the next state,
            reward - the reward,
            done - whether a terminal state was reached,
            info - any additional info"""
        assert self.action_space.contains(action), "%r (%s) is an invalid action" % (action, type(action))

        old_bbox = self.bbox
        self.action_set[action]()

        reward = self.compute_reward(old_bbox)

        self.history.insert(0, self.to_one_hot(action))

        if len(self.history) > TextLocEnv.HISTORY_LENGTH:
            self.history.pop()

        self.state = self.compute_state(), self.history

        return self.state, reward, self.done, {}

    def compute_reward(self, old_bbox):
        return 0

    def up(self):
        self.adjust_bbox(np.array([0, -1, 0, -1]))

    def down(self):
        self.adjust_bbox(np.array([0, 1, 0, 1]))

    def left(self):
        self.adjust_bbox(np.array([-1, 0, -1, 0]))

    def right(self):
        self.adjust_bbox(np.array([1, 0, 1, 0]))

    def zoom_in(self):
        self.adjust_bbox(np.array([1, 1, -1, -1]))

    def zoom_out(self):
        self.adjust_bbox(np.array([-1, -1, 1, 1]))

    def wider(self):
        self.adjust_bbox(np.array([-1, 0, 1, 0]))

    def taller(self):
        self.adjust_bbox(np.array([0, -1, 0, 1]))

    def trigger(self):
        self.done = True

    def adjust_bbox(self, directions):
        ah = round(self.alpha * (self.bbox[3] - self.bbox[1]))
        aw = round(self.alpha * (self.bbox[2] - self.bbox[0]))

        adjustments = np.array([aw, ah, aw, ah])
        delta = directions * adjustments

        self.bbox = self.bbox + delta

    def reset(self):
        """Reset the environment to its initial state (the bounding box covers the entire image"""
        self.history = []
        self.bbox = np.array([0, 0, self.image.width, self.image.height])
        self.state = self.compute_state()
        self.done = False

        return self.state

    def render(self, mode='human'):
        """Render the current state"""
        copy = self.image.copy()
        draw = ImageDraw.Draw(copy)
        draw.rectangle(self.bbox.tolist(), outline=(255, 255, 255))
        copy.show()

    def get_warped_bbox_contents(self):
        croppped = self.image.crop(self.bbox)
        return croppped.resize((224, 224), LANCZOS)

    def compute_state(self):
        return self.extract_features(), self.history

    def extract_features(self):
        """Compute the state from the image, the bounding box and the action history"""
        warped = self.get_warped_bbox_contents()
        feature = self.feature_extractor.extract([warped], layers=["fc7"])["fc7"]

        return feature[0]

    def to_one_hot(self, action):
        line = np.zeros(self.action_space.n, np.bool)
        line[action] = 1

        return line
