import numpy
from chainer.backends import cuda


class ImageMasker:
    """
        This class implements a CPU/GPU method for adding a mask on an array. Think of it as drawing a polygon that is defined
        by a bounding box, given by four points.
    """

    def __init__(self, fill_value):
        """
            :param fill_value: the value to fill the array with (if you want a black image, take 0)
        """
        self.fill_value = fill_value

    def create_mask_cpu(self, base_mask, corners):
        top_width_vectors = corners[:, 1, :] - corners[:, 0, :]
        bottom_width_vectors = corners[:, 3, :] - corners[:, 2, :]
        left_height_vectors = corners[:, 0, :] - corners[:, 3, :]
        right_height_vectors = corners[:, 2, :] - corners[:, 1, :]

        for idx in numpy.ndindex(base_mask.shape):
            h, w, batch_index = idx

            top_width_vector = top_width_vectors[batch_index]
            bottom_width_vector = bottom_width_vectors[batch_index]
            left_height_vector = left_height_vectors[batch_index]
            right_height_vector = right_height_vectors[batch_index]

            # determine cross product of each vector with our current point
            cross_products = []
            for corner, vector in zip(corners[batch_index], [top_width_vector, right_height_vector, bottom_width_vector, left_height_vector]):
                cross_product = vector[0] * (h - corner[1]) - vector[1] * (w - corner[0])
                cross_products.append(cross_product >= 0)

            # if the point is on the right of all lines (i.e positive) the point is inside the box and we can mark it
            if all(cross_products):
                base_mask[idx] = self.fill_value

        return base_mask

    def create_mask_gpu(self, base_mask, corners):
        create_mask_kernel = cuda.cupy.ElementwiseKernel(
            'T originalMask, raw T corners, int32 inputHeight, int32 batchCount, T fillValue',
            'T mask',
            '''
                // determine our current position in the array
                int batchIndex = i % batchCount;
                int w = (i / batchCount) % inputHeight;
                int h = i / batchCount / inputHeight;
                int cornerIndex = batchIndex * 4 * 2;

                // calculate vectors for each side of the box
                int2 topVec = {
                    corners[cornerIndex + 1 * 2] - corners[cornerIndex],
                    corners[cornerIndex + 1 * 2 + 1] - corners[cornerIndex + 1]
                };

                int2 bottomVec = {
                    corners[cornerIndex + 3 * 2] - corners[cornerIndex + 2 * 2],
                    corners[cornerIndex + 3 * 2 + 1] - corners[cornerIndex + 2 * 2 + 1]
                };

                int2 leftVec = {
                    corners[cornerIndex] - corners[cornerIndex + 3 * 2],
                    corners[cornerIndex + 1] - corners[cornerIndex + 3 * 2 + 1]
                }; 

                int2 rightVec = {
                    corners[cornerIndex + 2 * 2] - corners[cornerIndex + 1 * 2],
                    corners[cornerIndex + 2 * 2 + 1] - corners[cornerIndex + 1 * 2 + 1]
                };

                // calculate cross product for each side of array
                int crossTop = topVec.x * (h - corners[cornerIndex + 1]) - topVec.y * (w - corners[cornerIndex]);
                int crossRight = rightVec.x * (h - corners[cornerIndex + 3]) - rightVec.y * (w - corners[cornerIndex + 2]);
                int crossBottom = bottomVec.x * (h - corners[cornerIndex + 5]) - bottomVec.y * (w - corners[cornerIndex + 4]);
                int crossLeft = leftVec.x * (h - corners[cornerIndex + 7]) - leftVec.y * (w - corners[cornerIndex + 6]);

                // our point is inside as long as every cross product is greater or equal to 0
                bool inside = crossTop >= 0 && crossRight >= 0 && crossBottom >= 0 && crossLeft >= 0;

                mask = inside ? fillValue : originalMask;
            ''',
            name='bbox_to_mask',
        )

        height, width, channels = base_mask.shape
        mask = create_mask_kernel(base_mask, corners, height, channels, self.fill_value)
        return mask
        
    def mask_array(self, array, corners, xp):
        """
        Create a mask on a given array, using the OOB specified by corners

        :param array: An array with three dimensions. The first dimension is the width of the feature map or image.
        The second dimension is the height of the feature map or image. The third dimension is the batch dimension
        (if you want to mask a 4 dimensional array, you'll first need to collapse the batch and channel dimension, but
        make sure to also pad the given corners!, that means increase the number of corners to match
        batch size * number_of_channels).

        :param corners: An array with three dimensions. The first dimension is the batch axis (similar to the array
        parameter). The shape of the second dimension must be 4, as we are using bounding boxes with four corner points.
        The points of the bounding box must be in clockwise order.The shape of the third dimension (the actual points
        of each corner) should be 2. The first element in this dimension must be the x coordinate and the second
        element must be the y coordinate.

        :param xp: either numpy or cupy, depending on whether the code should run on CPU or GPU

        :return: an array where all elements inside the bounding box get the value that is saved in `self.fill_value`.
        """
        if xp == cuda.cupy:
            return self.create_mask_gpu(array, corners)
        else:
            return self.create_mask_cpu(array, corners)
