import torch


def translate(x, y, z, dtype=torch.float32):
    return torch.tensor([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]], dtype=dtype)


def scale(x, y, z, dtype=torch.float32):
    # scaling the image changes the size of the voxels, so we need to adjust the origin to keep the edges of the image
    # consistent with the original image
    return torch.tensor([[x, 0, 0, (x - 1)/2],
                         [0, y, 0, (y - 1)/2],
                         [0, 0, z, (z - 1)/2],
                         [0, 0, 0, 1]], dtype=dtype)


def rotate(theta, shape, dtype=torch.float32):
    # we want to rotate around the center of the image, so we need to translate the origin to the center of the image

    # translate the origin to the center of the image
    center = translate(*-1*((shape-1)/2), dtype=dtype)
    # This was a bit confusing to me at first, but we need to subtract 1 from the shape because the origin of the
    # image is the center of the corner voxel, not the corner of the corner voxel.  So, if the image is 100x100x100,
    # the distance between the origin and the center of the opposite corner is 99 voxels, not 100. Therefore,
    # the center of the image is at (99/2, 99/2, 99/2), not (100/2, 100/2, 100/2).

    # rotate
    rotate = rotate_x(theta[0], dtype=dtype) @ rotate_y(theta[1], dtype=dtype) @ rotate_z(theta[2], dtype=dtype)

    # translate the origin back to the corner of the image
    goback = translate(*((shape-1)/2), dtype=dtype)

    return goback @ rotate @ center


def rotate_x(theta, dtype=torch.float32):
    return torch.tensor([[1, 0, 0, 0],
                         [0, torch.cos(theta), -torch.sin(theta), 0],
                         [0, torch.sin(theta), torch.cos(theta), 0],
                         [0, 0, 0, 1]], dtype=dtype)


def rotate_y(theta, dtype=torch.float32):
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta), 0],
                         [0, 1, 0, 0],
                         [-torch.sin(theta), 0, torch.cos(theta), 0],
                         [0, 0, 0, 1]], dtype=dtype)


def rotate_z(theta, dtype=torch.float32):
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0, 0],
                         [torch.sin(theta), torch.cos(theta), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=dtype)


def shear(params, shape, dtype=torch.float32):
    shape = torch.tensor(shape, dtype=torch.float32)
    # translate the origin to the center of the image
    center = translate(*-1 * ((shape - 1) / 2), dtype=dtype)

    # shear
    affine = torch.tensor([[1, params[0], params[1], 0],
                         [params[2], 1, params[3], 0],
                         [params[4], params[5], 1, 0],
                         [0, 0, 0, 1]], dtype=dtype)

    # translate the origin back to the corner of the image
    goback = translate(*((shape - 1) / 2), dtype=dtype)

    return goback @ affine @ center


def flip(flip_params, shape):
    """
    :param flip_params: boolean list of length 3 indicating whether to flip the image along each axis
    :param shape: shape of the image
    :return: affine matrix to flip the image
    """
    translate_params = [0, 0, 0]
    scale_params = [1, 1, 1]
    if flip_params[0]:
        # Note: I translate by the shape instead of shape - 1
        # because the issues with the voxel coordinates are taken care of in the scaling function
        translate_params[0] = shape[0]
        scale_params[0] = -1
    if flip_params[1]:
        translate_params[1] = shape[1]
        scale_params[1] = -1
    if flip_params[2]:
        translate_params[2] = shape[2]
        scale_params[2] = -1
    return translate(*translate_params) @ scale(*scale_params)


