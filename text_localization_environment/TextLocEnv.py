import gym
from gym import spaces
from gym.utils import seeding
from chainer.backends import cuda
from PIL import Image, ImageDraw
from PIL.Image import LANCZOS, MAX_IMAGE_PIXELS
import numpy as np
from text_localization_environment.ImageMasker import ImageMasker


class TextLocEnv(gym.Env):

    HISTORY_LENGTH = 10
    # ⍺: factor relative to the current box size that is used for every transformation action
    ALPHA = 0.2
    # τ: Threshold of intersection over union for the trigger action to yield a positive reward
    TAU = 0.6
    # η: Reward of the trigger action
    ETA = 10.0

    def __init__(self, image_paths, true_bboxes, gpu_id=-1):
        """
        :param image_paths: The paths to the individual images
        :param true_bboxes: The true bounding boxes for each image
        :param gpu_id: The ID of the GPU to be used. -1 if CPU should be used instead
        :type image_paths: String or list
        :type true_bboxes: numpy.ndarray
        :type gpu_id: int
        """
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

        self.gpu_id = gpu_id
        if type(image_paths) is not list: image_paths = [image_paths]
        self.image_paths = image_paths
        self.true_bboxes = true_bboxes

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Execute an action and return
            state - the next state,
            reward - the reward,
            done - whether a terminal state was reached,
            info - any additional info"""
        assert self.action_space.contains(action), "%r (%s) is an invalid action" % (action, type(action))

        self.action_set[action]()

        reward = self.calculate_reward(action)
        self.max_iou = max(self.iou, self.max_iou)

        self.history.insert(0, self.to_one_hot(action))
        self.history.pop()

        self.state = self.compute_state()

        info = self.find_positive_actions()

        return self.state, reward, self.done, info

    def calculate_reward(self, action):
        reward = 0

        if self.action_set[action] == self.trigger:
            if self.iou >= self.TAU:
                reward = self.ETA
            else:
                reward = -self.ETA
        else:
            new_iou = self.compute_best_iou()
            reward = np.sign(new_iou - self.iou)

            if reward == 0:
                self.steps_since_last_change += 1
            else:
                self.steps_since_last_change = 0

            if self.steps_since_last_change >= 3:
                reward = -1

            self.iou = new_iou

        return reward

    def calculate_potential_reward(self, action):
        old_bbox = self.bbox
        old_iou = self.iou

        if self.action_set[action] != self.trigger:
            self.action_set[action]()

        reward = self.calculate_reward(action)

        self.bbox = old_bbox
        self.iou = old_iou

        return reward

    def find_positive_actions(self):
        rewards = np.array([self.calculate_potential_reward(i) for i in self.action_set])

        positive_actions = np.arange(0, self.action_space.n)[rewards > 0]

        if len(positive_actions) == 0:
            return np.arange(0, self.action_space.n).tolist()

        return positive_actions.tolist()

    def create_empty_history(self):
        flat_history = np.repeat([False], self.HISTORY_LENGTH * self.action_space.n)
        history = flat_history.reshape((self.HISTORY_LENGTH, self.action_space.n))

        return history.tolist()

    @staticmethod
    def to_four_corners_array(two_bbox):
        """
        Creates an array of bounding boxes with four corners out of a bounding box with two corners, so
        that the ImageMasker can be applied.

        :param two_bbox: Bounding box with two points, top left and bottom right

        :return: An array of bounding boxes that corresponds to the requirements of the ImageMasker
        """
        top_left = np.array([two_bbox[0], two_bbox[1]], dtype=np.int32)
        bottom_left = np.array([two_bbox[0], two_bbox[3]], dtype=np.int32)
        top_right = np.array([two_bbox[2], two_bbox[1]], dtype=np.int32)
        bottom_right = np.array([two_bbox[2], two_bbox[3]], dtype=np.int32)

        four_bbox = np.array([bottom_right, bottom_left, top_left, top_right])

        return np.array([four_bbox, four_bbox, four_bbox])

    def create_ior_mark(self):
        """
        Creates an IoR (inhibition of return) mark that crosses out the current bounding box.
        This is necessary to find multiple objects within one image
        """
        masker = ImageMasker(0)

        center_height = round((self.bbox[3] + self.bbox[1]) / 2)
        center_width = round((self.bbox[2] + self.bbox[0]) / 2)
        height_frac = round((self.bbox[3] - self.bbox[1]) / 12)
        width_frac = round((self.bbox[2] - self.bbox[0]) / 12)

        horizontal_box = [self.bbox[0], center_height - height_frac, self.bbox[2], center_height + height_frac]
        vertical_box = [center_width - width_frac, self.bbox[1], center_width + width_frac, self.bbox[3]]

        horizontal_box_four_corners = self.to_four_corners_array(horizontal_box)
        vertical_box_four_corners = self.to_four_corners_array(vertical_box)

        array_module = np

        if self.gpu_id != -1:
            array_module = cuda.cupy
            horizontal_box_four_corners = cuda.to_gpu(horizontal_box_four_corners, self.gpu_id)
            vertical_box_four_corners = cuda.to_gpu(vertical_box_four_corners, self.gpu_id)

        new_img = array_module.array(self.episode_image, dtype=np.int32)
        new_img = masker.mask_array(new_img, horizontal_box_four_corners, array_module)
        new_img = masker.mask_array(new_img, vertical_box_four_corners, array_module)

        if self.gpu_id != -1:
            self.episode_image = Image.fromarray(cuda.to_cpu(new_img).astype(np.uint8))
        else:
            self.episode_image = Image.fromarray(new_img.astype(np.uint8))

    def compute_best_iou(self):
        max_iou = 0

        for box in self.episode_true_bboxes:
            max_iou = max(max_iou, self.compute_iou(box))

        return max_iou

    def compute_iou(self, other_bbox):
        """Computes the intersection over union of the argument and the current bounding box."""
        intersection = self.compute_intersection(other_bbox)

        area_1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        area_2 = (other_bbox[1][0] - other_bbox[0][0]) * (other_bbox[1][1] - other_bbox[0][1])
        union = area_1 + area_2 - intersection

        return intersection / union

    def compute_intersection(self, other_bbox):
        left = max(self.bbox[0], other_bbox[0][0])
        top = max(self.bbox[1], other_bbox[0][1])
        right = min(self.bbox[2], other_bbox[1][0])
        bottom = min(self.bbox[3], other_bbox[1][1])

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
        self.adjust_bbox(np.array([-0.5, -0.5, 0.5, 0.5]))

    def smaller(self):
        self.adjust_bbox(np.array([0.5, 0.5, -0.5, -0.5]))

    def fatter(self):
        self.adjust_bbox(np.array([-1, 0, 1, 0]))

    def taller(self):
        self.adjust_bbox(np.array([0, -1, 0, 1]))

    def trigger(self):
        self.done = True
        self.create_ior_mark()

    @staticmethod
    def box_size(box):
        width = box[2] - box[0]
        height = box[3] - box[1]

        return width * height

    def adjust_bbox(self, directions):
        ah = round(self.ALPHA * (self.bbox[3] - self.bbox[1]))
        aw = round(self.ALPHA * (self.bbox[2] - self.bbox[0]))

        adjustments = np.array([aw, ah, aw, ah])
        delta = directions * adjustments

        new_box = self.bbox + delta

        if self.box_size(new_box) < MAX_IMAGE_PIXELS:
            self.bbox = new_box

    def reset(self):
        """Reset the environment to its initial state (the bounding box covers the entire image"""
        self.history = self.create_empty_history()

        random_index = self.np_random.randint(len(self.image_paths))
        self.episode_image = Image.open(self.image_paths[random_index])

        if self.episode_image.mode != 'RGB':
            self.episode_image = self.episode_image.convert('RGB')

        self.episode_true_bboxes = self.true_bboxes[random_index]

        self.bbox = np.array([0, 0, self.episode_image.width, self.episode_image.height])
        self.state = self.compute_state()
        self.done = False
        self.iou = self.compute_best_iou()
        self.max_iou = self.iou
        self.steps_since_last_change = 0

        return self.state

    def render(self, mode='human'):
        """Render the current state"""

        if mode == 'human':
            copy = self.episode_image.copy()
            draw = ImageDraw.Draw(copy)
            draw.rectangle(self.bbox.tolist(), outline=(255, 255, 255))
            copy.show()
        elif mode == 'box':
            warped = self.get_warped_bbox_contents()
            warped.show()

    def get_warped_bbox_contents(self):
        cropped = self.episode_image.crop(self.bbox)
        return cropped.resize((224, 224), LANCZOS)

    def compute_state(self):
        image_array = np.array(self.get_warped_bbox_contents(), dtype=np.float32).transpose((2, 0, 1))
        history = np.array(self.history, dtype=np.float32).flatten()

        return image_array  # , history

    def to_one_hot(self, action):
        line = np.zeros(self.action_space.n, np.bool)
        line[action] = 1

        return line
