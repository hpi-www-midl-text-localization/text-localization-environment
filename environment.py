import gym
from gym import spaces
from chainer.links import VGG16Layers
from PIL import ImageDraw
from PIL.Image import LANCZOS
import numpy as np


class TextLocEnv(gym.Env):

    HISTORY_LENGTH = 10
    # ⍺: factor relative to the current box size that is used for every transformation action
    ALPHA = 0.2
    # τ: Threshold of intersection over union for the trigger action to yield a positive reward
    TAU = 0.6
    # η: Reward of the trigger action
    ETA = 3.0

    def __init__(self, image, true_bboxes, use_gpu=False):
        self.feature_extractor = VGG16Layers()
        self.action_space = spaces.Discrete(9)
        self.action_set = {0: self.right,
                           1: self.left,
                           2: self.up,
                           3: self.down,
                           4: self.bigger,
                           5: self.smaller,
                           6: self.fatter,
                           7: self.taller,
                           8: self.trigger
                           }

        if use_gpu:
            self.feature_extractor.to_gpu(0)

        self.image = image
        self.true_bboxes = true_bboxes
        self.history = self.create_empty_history()
        self.bbox = np.array([0, 0, image.width, image.height])
        self.iou = 0
        self.state = self.compute_state()
        self.done = False

    def step(self, action):
        """Execute an action and return
            state - the next state,
            reward - the reward,
            done - whether a terminal state was reached,
            info - any additional info"""
        assert self.action_space.contains(action), "%r (%s) is an invalid action" % (action, type(action))

        self.action_set[action]()

        reward = 0
        if self.action_set[action] == self.trigger:
            if self.iou >= self.TAU:
                reward = self.ETA
            else:
                reward = -self.ETA
        else:
            new_iou = self.compute_best_iou()
            reward = np.sign(new_iou - self.iou)
            self.iou = new_iou

        self.history.insert(0, self.to_one_hot(action))
        self.history.pop()

        self.state = self.compute_state()

        return self.state, reward, self.done, {}

    def create_empty_history(self):
        flat_history = np.repeat([False], self.HISTORY_LENGTH * self.action_space.n)
        history = flat_history.reshape((self.HISTORY_LENGTH, self.action_space.n))

        return history.tolist()

    def compute_best_iou(self):
        max_iou = 0

        for box in self.true_bboxes:
            max_iou = max(max_iou, self.compute_iou(box))

        return max_iou

    def compute_iou(self, other_bbox):
        """Computes the intersection over union of the argument and the current bounding box."""
        intersection = self.compute_intersection(other_bbox)

        area_1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        area_2 = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
        union = area_1 + area_2 - intersection

        return intersection / union

    def compute_intersection(self, other_bbox):
        left = max(self.bbox[0], other_bbox[0])
        top = max(self.bbox[1], other_bbox[1])
        right = min(self.bbox[2], other_bbox[2])
        bottom = min(self.bbox[3], other_bbox[3])

        if right < left or bottom < top:
            return 0

        return (right - left) * (bottom - top)

    def up(self):
        self.adjust_bbox(np.array([0, -1, 0, -1]))

    def down(self):
        self.adjust_bbox(np.array([0, 1, 0, 1]))

    def left(self):
        self.adjust_bbox(np.array([-1, 0, -1, 0]))

    def right(self):
        self.adjust_bbox(np.array([1, 0, 1, 0]))

    def bigger(self):
        self.adjust_bbox(np.array([1, 1, -1, -1]))

    def smaller(self):
        self.adjust_bbox(np.array([-1, -1, 1, 1]))

    def fatter(self):
        self.adjust_bbox(np.array([-1, 0, 1, 0]))

    def taller(self):
        self.adjust_bbox(np.array([0, -1, 0, 1]))

    def trigger(self):
        self.done = True

    def adjust_bbox(self, directions):
        ah = round(self.ALPHA * (self.bbox[3] - self.bbox[1]))
        aw = round(self.ALPHA * (self.bbox[2] - self.bbox[0]))

        adjustments = np.array([aw, ah, aw, ah])
        delta = directions * adjustments

        new_box = self.bbox + delta
        new_box[0] = max(new_box[0], 0)
        new_box[1] = max(new_box[1], 0)
        new_box[2] = min(new_box[2], self.image.width)
        new_box[3] = min(new_box[3], self.image.height)

        self.bbox = new_box

    def reset(self):
        """Reset the environment to its initial state (the bounding box covers the entire image"""
        self.history = self.create_empty_history()
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
        cropped = self.image.crop(self.bbox)
        return cropped.resize((224, 224), LANCZOS)

    def compute_state(self):
        return np.concatenate((self.extract_features().array, np.array(self.history).flatten()))

    def extract_features(self):
        """Extract features from the image using the VGG16 network"""
        warped = self.get_warped_bbox_contents()
        feature = self.feature_extractor.extract([warped], layers=["fc7"])["fc7"]

        return feature[0]

    def to_one_hot(self, action):
        line = np.zeros(self.action_space.n, np.bool)
        line[action] = 1

        return line
