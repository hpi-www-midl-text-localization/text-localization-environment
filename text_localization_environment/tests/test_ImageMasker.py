import numpy as np
from PIL import Image, ImageDraw
from text_localization_environment import TextLocEnv
from text_localization_environment.ImageMasker import ImageMasker


def test_mask_array_cpu():
    masker = ImageMasker(0)
    image = Image.new("RGB", (4, 4), color=(255, 255, 255))

    masking_box = [1, 1, 2, 2]

    array_module = np
    actual_image = np.array(image)
    actual_image = masker.mask_array(actual_image, TextLocEnv.to_four_corners_array(masking_box), array_module)

    expected_image = Image.new("RGB", (4, 4), color=(255, 255, 255))
    draw = ImageDraw.Draw(expected_image)
    draw.rectangle(masking_box, fill=0)
    del draw
    expected_image = np.array(expected_image)

    assert ((actual_image == expected_image).all())


if __name__ == "__main__":
    test_mask_array_cpu()
