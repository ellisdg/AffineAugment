import torch


def augment(affine, translate_params=None, scale_params=None, rotate_params=None, shear_params=None, flip_params=None,
            shape=None):
    transforms = create_augmentation_transforms(translate_params, scale_params, rotate_params, shear_params,
                                                flip_params, shape)
    augmentation_transform = torch.eye(4, dtype=affine.dtype)
    for transform in transforms:
        augmentation_transform = torch.matmul(transform, augmentation_transform)
    return torch.matmul(affine, augmentation_transform)


def create_augmentation_transforms(translate_params=None, scale_params=None, rotate_params=None, shear_params=None,
                                   flip_params=None, shape=None):
    transforms = list()
    if translate_params is not None:
        transforms.append(translate(*translate_params))
    if scale_params is not None:
        transforms.append(scale(*scale_params))
    if rotate_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if rotate_params are provided")
        transforms.append(rotate(rotate_params, torch.tensor(shape)))
    if shear_params is not None:
        transforms.append(shear_x(*shear_params[0]))
        transforms.append(shear_y(*shear_params[1]))
        transforms.append(shear_z(*shear_params[2]))
    if flip_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if flip_params are provided")
        transforms.append(flip(flip_params, shape))
    return transforms


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


def shear_x(y, z, dtype=torch.float32):
    return torch.tensor([[1, y, z, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=dtype)


def shear_y(x, z, dtype=torch.float32):
    return torch.tensor([[1, 0, 0, 0],
                         [x, 1, z, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=dtype)


def shear_z(x, y, dtype=torch.float32):
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [x, y, 1, 0],
                         [0, 0, 0, 1]], dtype=dtype)


def flip(flip_params, shape):
    """
    :param flip_params: boolean list of length 3 indicating whether to flip the image along each axis
    :param shape: shape of the image
    :return: affine matrix to flip the image
    """
    translate_params = [0, 0, 0]
    scale_params = [1, 1, 1]
    if flip_params[0]:
        translate_params[0] = shape[0]
        scale_params[0] = -1
    if flip_params[1]:
        translate_params[1] = shape[1]
        scale_params[1] = -1
    if flip_params[2]:
        translate_params[2] = shape[2]
        scale_params[2] = -1
    return translate(*translate_params) @ scale(*scale_params)


