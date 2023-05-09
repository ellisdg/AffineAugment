import torch
from monai.transforms import SpatialResample, Affine, Zoom, Rotate, Flip, ResampleToMatch
from monai.data import MetaTensor
from affine import translate, scale, rotate, shear, flip


def resample(image, target_affine, target_shape, mode='bilinear', dtype=None, align_corners=False):
    if dtype:
        image = image.to(dtype)
    else:
        dtype = image.dtype

    resampler = ResampleToMatch(mode=mode, align_corners=align_corners, dtype=dtype)
    image.meta['filename_or_obj'] = "placeholder.nii.gz"
    return resampler(img=image, img_dst=MetaTensor(x=torch.zeros(target_shape, dtype=dtype),
                                                   affine=target_affine,
                                                   meta={'filename_or_obj': "placeholder.nii.gz"}))


def augment_image(image, translate_params=None, rotate_params=None, shear_params=None,flip_params=None,
                  scale_params=None, shape=None, verbose=False):
    affine = augment_affine(image.affine, translate_params=translate_params, rotate_params=rotate_params,
                            shear_params=shear_params, flip_params=flip_params, scale_params=scale_params,
                            shape=torch.tensor(shape), verbose=verbose)

    return resample(image, affine, image.shape)


def monai_augment_image(image, translate_params=None, rotate_params=None, shear_params=None, flip_params=None,
                        scale_params=None, keep_size=True):
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
        image = Zoom((1/scale_params).tolist(), keep_size=keep_size)(image)
    return image


def augment_affine(affine, translate_params=None, rotate_params=None, shear_params=None, flip_params=None,
                   scale_params=None, shape=None, verbose=False):
    transforms = create_augmentation_transforms(translate_params=translate_params,
                                                rotate_params=rotate_params,
                                                shear_params=shear_params,
                                                flip_params=flip_params,
                                                scale_params=scale_params,
                                                shape=shape)
    augmentation_transform = torch.eye(4, dtype=affine.dtype)
    for transform in transforms:
        augmentation_transform = torch.matmul(augmentation_transform, transform)
    if verbose:
        print(augmentation_transform)
    return torch.matmul(affine, augmentation_transform)


def create_augmentation_transforms(translate_params=None, rotate_params=None, shear_params=None,
                                   flip_params=None, scale_params=None, shape=None):
    if type(shape) == tuple:
        shape = torch.tensor(shape)
    transforms = list()
    if translate_params is not None:
        transforms.append(translate(*translate_params))
    if rotate_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if rotate_params are provided")
        transforms.append(rotate(rotate_params, shape))
    if shear_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if shear_params are provided")
        transforms.append(shear(shear_params, shape))
    if flip_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if flip_params are provided")
        transforms.append(flip(flip_params, shape))
    if scale_params is not None:
        if shape is None:
            raise ValueError("shape must be provided if scale_params are provided")
        transforms.append(scale(*scale_params, shape=shape))
    return transforms
