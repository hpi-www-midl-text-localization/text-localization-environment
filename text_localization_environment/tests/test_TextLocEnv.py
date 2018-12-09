import pytest
import numpy as np
from text_localization_environment import TextLocEnv


def test_to_four_corners_array():
    two_corners_aabb = np.array([0, 0, 20, 10])
    four_corners_array = TextLocEnv.to_four_corners_array(two_corners_aabb)
    assert((four_corners_array == np.array([np.array([np.array([20, 10]),
                                                      np.array([0, 10]),
                                                      np.array([0, 0]),
                                                      np.array([20, 0])]),
                                            np.array([np.array([20, 10]),
                                                      np.array([0, 10]),
                                                      np.array([0, 0]),
                                                      np.array([20, 0])]),
                                            np.array([np.array([20, 10]),
                                                      np.array([0, 10]),
                                                      np.array([0, 0]),
                                                      np.array([20, 0])])])).all())


if __name__ == "__main__":
    test_to_four_corners_array()
