"""
Compare the augmentation methods between my implementation which combines the transformations and performs a single
resampling step and the MONAI implementation which performs the transformations separately and performs a resampling
step for each transformation.
(Note: The Affine transformation does the same thing as my implementation, but I didn't know about it when I started
this project)
"""
from augment import augment_image, monai_augment_image, resample
import nibabel as nib
from monai.data import MetaTensor
import torch
from monai.utils import set_determinism
from monai.transforms import ResampleToMatch
import time
from random import shuffle
import os


def generate_random_parameters():
    translate_params = torch.randn(3) * 10
    # generate rotation parameters with mean 0 and std 0.5
    rotate_params = torch.randn(3) * 0.5
    # generate flip parameters
    flip_params = torch.rand(3) > 0.5
    # generate shear params with mean 0 and std 0.1
    shear_params = torch.randn(6) * 0.1
    # generate scale params with mean 1 and std 0.1
    scale_params = torch.randn(3) * 0.1 + 1
    params = {"translate_params": translate_params,
              "rotate_params": rotate_params,
              "flip_params": flip_params,
              "shear_params": shear_params,
              "scale_params": scale_params}
    return params


def generate_n_random_parameters(n):
    params = generate_random_parameters()
    param_keys = list(params.keys())
    # randomly select n parameter keys
    shuffle(param_keys)
    selected_param_keys = param_keys[:n]
    selected_params = dict()
    for key in selected_param_keys:
        selected_params[key] = params[key]
    return selected_params


def get_mean_squared_error(image1, image2):
    return torch.mean((image1 - image2) ** 2)


def get_signal_loss(image1, image2):
    # resample image2 back to image1's grid
    image2_resampled = ResampleToMatch()(image2, image1)
    # calculate the signal loss
    return get_mean_squared_error(image1, image2_resampled)


def run_augmentation_test(image, func, params, **kwargs):
    # augment the image
    start = time.time()
    augmented_image = func(image.detach().clone(), **params, **kwargs)
    stop = time.time()
    # calculate the signal loss
    signal_loss = get_signal_loss(image, augmented_image)
    return signal_loss.numpy(), stop - start, augmented_image


def run_augmentation_tests(image, params):
    my_signal_loss, my_time, my_augmented_image = run_augmentation_test(image, augment_image, params,
                                                                        shape=image.shape[1:])
    monai_signal_loss, monai_time, monai_augmented_image = run_augmentation_test(image, monai_augment_image, params)
    return my_signal_loss, monai_signal_loss, my_time, monai_time


def main():
    set_determinism(seed=0)
    T2w_image = nib.load("data/T2w_acpc_dc_restore_brain.nii.gz")
    T2w_data = MetaTensor(torch.from_numpy(T2w_image.get_fdata()[None]).to(dtype=torch.float32),
                          affine=torch.from_numpy(T2w_image.affine).to(dtype=torch.float32))
    T2w_data.meta["filename_or_obj"] = os.path.abspath("data/T2w_acpc_dc_restore_brain.nii.gz")

    results = list()
    for n_params in range(1, 6):
        print(f"Running tests with {n_params} parameters")
        for i in range(50):
            params = generate_n_random_parameters(n_params)
            my_signal_loss, monai_signal_loss, my_time, monai_time = run_augmentation_tests(T2w_data, params)
            print(list(params.keys()))
            print(f"My signal loss: {my_signal_loss:.4f}, MONAI signal loss: {monai_signal_loss:.4f}")
            print(f"My time: {my_time:.4f}, MONAI time: {monai_time:.4f}")
            results.append([n_params, list(params.keys()), my_signal_loss, monai_signal_loss, my_time, monai_time])
    with open("results/signal_loss_and_transformation_time.tsv", "w") as f:
        f.write("n_params\tparams\tmy_signal_loss\tmonai_signal_loss\tmy_time\tmonai_time\n")
        for result in results:
            f.write("\t".join(map(str, result)) + "\n")


if __name__ == "__main__":
    with torch.no_grad():
        main()
