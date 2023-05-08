import torch
from monai.transforms import SpatialResample, Affine, Zoom, Rotate, Flip

from affine import translate, scale, rotate, shear, flip


def resample(image, target_affine, target_shape, mode='bilinear', dtype=None, align_corners=True):
    if dtype:
        image = image.to(dtype)
    else:
        dtype = image.dtype

    resampler = SpatialResample(mode=mode, align_corners=align_corners, dtype=dtype)

    return resampler(img=image, dst_affine=target_affine, spatial_size=target_shape)


def augment_image(image, translate_params=None, rotate_params=None, shear_params=None,flip_params=None,
                  scale_params=None, shape=None):
    affine = augment_affine(image.affine, translate_params=translate_params, rotate_params=rotate_params,
                            shear_params=shear_params, flip_params=flip_params, scale_params=scale_params, shape=shape)

    return resample(image, affine, image.shape)


def monai_augment_image(image, translate_params=None, rotate_params=None, shear_params=None, flip_params=None,
                        scale_params=None,):
    if translate_params is not None:
        image, _ = Affine(translate_params=translate_params.tolist())(image.detach().clone())
    if rotate_params is not None:
        image = Rotate(angle=rotate_params)(image)
    if shear_params is not None:
        image, _ = Affine(shear_params=shear_params.tolist())(image.detach().clone())
    if flip_params is not None:
        for index in range(len(flip_params)):
            if flip_params[index]:
                image = Flip(index)(image)
    if scale_params is not None:
        image = Zoom((1/scale_params).tolist(), keep_size=False)(image)
    return image


def augment_affine(affine, translate_params=None, rotate_params=None, shear_params=None, flip_params=None,
                   scale_params=None, shape=None):
    transforms = create_augmentation_transforms(translate_params=translate_params,
                                                rotate_params=rotate_params,
                                                shear_params=shear_params,
                                                flip_params=flip_params,
                                                scale_params=scale_params,
                                                shape=shape)
    augmentation_transform = torch.eye(4, dtype=affine.dtype)
    for transform in transforms:
        augmentation_transform = torch.matmul(augmentation_transform, transform)
    return torch.matmul(affine, augmentation_transform)


def create_augmentation_transforms(translate_params=None, rotate_params=None, shear_params=None,
                                   flip_params=None, scale_params=None, shape=None):
    transforms = list()
    if translate_params is not None:
        transforms.append(translate(*translate_params))
    if rotate_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if rotate_params are provided")
        transforms.append(rotate(rotate_params, torch.tensor(shape)))
    if shear_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if shear_params are provided")
        transforms.append(shear(shear_params, shape))
    if flip_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if flip_params are provided")
        transforms.append(flip(flip_params, shape))
    if scale_params is not None:
        transforms.append(scale(*scale_params))
    return transforms
