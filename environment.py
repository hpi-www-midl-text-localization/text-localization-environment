import gym
from chainer.links import VGG16Layers
from PIL import ImageDraw
from PIL.Image import LANCZOS


class TextLocEnv(gym.Env):

    def __init__(self, image, true_bboxes):
        self.alpha = 0.2
        self.feature_extractor = VGG16Layers()
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
        self.bbox = (0, 0, image.width, image.height)
        self.state = self.compute_state()
        self.done = False

    def step(self, action):
        """Execute an action and return
            state - the next state,
            reward - the reward,
            done - whether a terminal state was reached,
            info - any additional info"""
        old_bbox = self.bbox
        self.action_set[action]()

        reward = self.compute_reward(old_bbox)

        self.state = self.compute_state()

        return self.state, reward, self.done, {}

    def compute_reward(self, old_bbox):
        return 0

    def up(self):
        pass

    def down(self):
        pass

    def left(self):
        pass

    def right(self):
        pass

    def zoom_in(self):
        pass

    def zoom_out(self):
        pass

    def wider(self):
        pass

    def taller(self):
        pass

    def trigger(self):
        pass

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

    def get_warped_bbox_contents(self):
        croppped = self.image.crop(self.bbox)
        return croppped.resize((224, 224), LANCZOS)

    def compute_state(self):
        """Compute the state from the image, the bounding box and the action history"""
        warped = self.get_warped_bbox_contents()
        feature = self.feature_extractor.extract([warped], layers=["fc7"])["fc7"]

        return feature[0]
