import torch
from augment import augment_image, monai_augment_image, resample
import nibabel as nib
from monai.data import MetaTensor
import time
from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt
import numpy as np


def view_data(image, output_file):
    spacing = image.header.get_zooms()
    shape = image.shape
    data = np.asarray(image.dataobj)
    sizex = shape[0] * spacing[0]
    sizey = shape[1] * spacing[1]
    sizez = shape[2] * spacing[2]
    norm_factor = np.max([sizex, sizey, sizez])
    sizex = (sizex / norm_factor) * 5
    sizey = (sizey / norm_factor) * 5
    sizez = (sizez / norm_factor) * 5
    fig, ax = plt.subplots(1, 1, figsize=(sizey, sizex))
    ax.imshow(data[:, :, shape[2] // 2], cmap="gray", aspect="auto")
    fig.savefig(output_file.replace(".png", "_xy.png"))
    fig, ax = plt.subplots(1, 1, figsize=(sizez, sizey))
    ax.imshow(data[shape[0] // 2, :, :], cmap="gray", aspect="auto")
    fig.savefig(output_file.replace(".png", "_yz.png"))
    fig, ax = plt.subplots(1, 1, figsize=(sizez, sizex))
    ax.imshow(data[:, shape[1] // 2, :], cmap="gray", aspect="auto")
    fig.savefig(output_file.replace(".png", "_xz.png"))


def main():
    T2w_image = nib.load("data/T2w_acpc_dc_restore_brain.nii.gz")
    T2w_data = MetaTensor(torch.from_numpy(T2w_image.get_fdata()[None]).to(dtype=torch.float32),
                          affine=torch.from_numpy(T2w_image.affine).to(dtype=torch.float32))
    view_data(T2w_image, output_file="data/examples/T2w/T2w_view.png")
    coords = (-15, -1, 38)  # coordinates for consistent plotting
    plot_anat(T2w_image, title="T2w", output_file="data/examples/T2w/T2w.png", cut_coords=coords)

    # example of a translation
    translate_params = torch.tensor([25, 25, 25])
    start = time.time()
    translated = augment_image(image=T2w_data.detach().clone(), translate_params=translate_params, rotate_params=None,
                               flip_params=None, shear_params=None, scale_params=None, shape=T2w_data.shape[1:],
                               verbose=True)
    stop = time.time()
    print("Translation took {} seconds".format(stop - start))
    translated = nib.Nifti1Image(translated.numpy()[0], translated.affine.numpy())
    translated.to_filename("data/examples/T2w/translated_{}_T2w.nii.gz".format("_".join(
        [str(x) for x in translate_params.tolist()])))
    view_data(translated, output_file="data/examples/T2w/translated_{}_T2w_view.png".format("_".join(
        [str(x) for x in translate_params.tolist()]))
              )
    plot_anat(translated, title="Translated T2w", cut_coords=coords,
              output_file="data/examples/T2w/translated_{}_T2w.png".format(
                  "_".join([str(x) for x in translate_params.tolist()])))

    # example of a monai translation
    start = time.time()
    translated = monai_augment_image(image=T2w_data.detach().clone(), translate_params=translate_params,
                                     rotate_params=None, flip_params=None, shear_params=None, scale_params=None,
                                     keep_size=False)
    stop = time.time()
    print("Monai translation took {} seconds".format(stop - start))
    translated = nib.Nifti1Image(translated.numpy()[0], translated.affine.numpy())
    translated.to_filename("data/examples/T2w/monai_translated_{}_T2w.nii.gz".format("_".join(
        [str(x) for x in translate_params.tolist()])))
    view_data(translated, output_file="data/examples/T2w/monai_translated_{}_T2w_view.png".format("_".join(
        [str(x) for x in translate_params.tolist()]))
              )
    plot_anat(translated, title="Monai Translated T2w", cut_coords=coords,
              output_file="data/examples/T2w/monai_translated_{}_T2w.png".format(
                  "_".join([str(x) for x in translate_params.tolist()]))
              )

    # example of a rotation
    rotation_params = torch.tensor([0, torch.pi / 10, 0])  # 18 degrees
    start = time.time()
    rotated = augment_image(image=T2w_data, translate_params=None, rotate_params=rotation_params,
                            flip_params=None, shear_params=None, scale_params=None, shape=T2w_data.shape[1:],
                            verbose=True)
    stop = time.time()
    print("Rotation took {} seconds".format(stop - start))

    rotated = nib.Nifti1Image(rotated.numpy()[0], rotated.affine.numpy())
    rotated.to_filename("data/examples/T2w/rotated_pi4_0_0_T2w.nii.gz")
    view_data(rotated, output_file="data/examples/T2w/rotated_pi4_0_0_T2w_view.png")
    plot_anat(rotated, title="Rotated T2w", cut_coords=coords,
              output_file="data/examples/T2w/rotated_pi4_0_0_T2w.png")

    # example of a monai rotation
    start = time.time()
    rotated = monai_augment_image(image=T2w_data, translate_params=None, rotate_params=rotation_params,
                                  flip_params=None, shear_params=None, scale_params=None, keep_size=False)
    stop = time.time()
    print("Monai rotation took {} seconds".format(stop - start))

    rotated = nib.Nifti1Image(rotated.numpy()[0], rotated.affine.numpy())
    rotated.to_filename("data/examples/T2w/monai_rotated_pi4_0_0_T2w.nii.gz")
    view_data(rotated, output_file="data/examples/T2w/monai_rotated_pi4_0_0_T2w_view.png")
    plot_anat(rotated, title="Monai Rotated T2w", output_file="data/examples/T2w/monai_rotated_pi4_0_0_T2w.png",
              cut_coords=coords)

    # example of a flip
    flip_params = torch.tensor([True, True, True])
    start = time.time()
    flipped = augment_image(image=T2w_data, translate_params=None, rotate_params=None,
                            flip_params=flip_params, shear_params=None, scale_params=None, shape=T2w_data.shape[1:],
                            verbose=True)
    stop = time.time()
    print("Flip took {} seconds".format(stop - start))
    flipped = nib.Nifti1Image(flipped.numpy()[0], flipped.affine.numpy())
    flipped.to_filename("data/examples/T2w/flipped_{}_{}_{}_T2w.nii.gz".format(*flip_params.tolist()))
    view_data(flipped, output_file="data/examples/T2w/flipped_{}_{}_{}_T2w_view.png".format(*flip_params.tolist()))
    plot_anat(flipped, title="Flipped T2w", cut_coords=coords,
              output_file="data/examples/T2w/flipped_{}_{}_{}_T2w.png".format(*flip_params.tolist()))

    # example of a monai flip
    start = time.time()
    flipped = monai_augment_image(image=T2w_data, translate_params=None, rotate_params=None,
                                  flip_params=flip_params, shear_params=None, scale_params=None, keep_size=False)
    stop = time.time()
    print("Monai flip took {} seconds".format(stop - start))
    flipped = nib.Nifti1Image(flipped.numpy()[0], flipped.affine.numpy())
    flipped.to_filename("data/examples/T2w/monai_flipped_{}_{}_{}_T2w.nii.gz".format(*flip_params.tolist()))
    view_data(flipped, output_file="data/examples/T2w/monai_flipped_{}_{}_{}_T2w_view.png".format(*flip_params.tolist()))
    plot_anat(flipped, title="Monai Flipped T2w", cut_coords=coords,
              output_file="data/examples/T2w/monai_flipped_{}_{}_{}_T2w.png".format(*flip_params.tolist()))

    # example of a shear
    shear_params = torch.tensor([0.2, 0.2, 0, 0, 0, 0])
    start = time.time()
    sheared = augment_image(image=T2w_data, translate_params=None, rotate_params=None,
                            flip_params=None, shear_params=shear_params, scale_params=None, shape=T2w_data.shape[1:],
                            verbose=True)
    stop = time.time()
    print("Shear took {} seconds".format(stop - start))
    sheared = nib.Nifti1Image(sheared.numpy()[0], sheared.affine.numpy())
    sheared.to_filename("data/examples/T2w/sheared_{}_{}_{}_T2w.nii.gz".format(*shear_params.tolist()))
    view_data(sheared, output_file="data/examples/T2w/sheared_{}_{}_{}_T2w_view.png".format(*shear_params.tolist()))
    plot_anat(sheared, title="Sheared T2w", cut_coords=coords,
              output_file="data/examples/T2w/sheared_{}_{}_{}_T2w.png".format(*shear_params.tolist()))

    # example of a monai shear
    start = time.time()
    sheared = monai_augment_image(image=T2w_data, translate_params=None, rotate_params=None,
                                  flip_params=None, shear_params=shear_params, scale_params=None)
    stop = time.time()
    print("Monai shear took {} seconds".format(stop - start))
    sheared = nib.Nifti1Image(sheared.numpy()[0], sheared.affine.numpy())
    sheared.to_filename("data/examples/T2w/monai_sheared_{}_{}_{}_T2w.nii.gz".format(*shear_params.tolist()))
    view_data(sheared, output_file="data/examples/T2w/monai_sheared_{}_{}_{}_T2w_view.png".format(*shear_params.tolist()))
    plot_anat(sheared, title="Monai Sheared T2w", cut_coords=coords,
              output_file="data/examples/T2w/monai_sheared_{}_{}_{}_T2w.png".format(*shear_params.tolist()))

    # example of a scale
    scale_params = torch.tensor([1.5, 0.5, 2.0])
    start = time.time()
    scaled = augment_image(image=T2w_data, translate_params=None, rotate_params=None,
                           flip_params=None, shear_params=None, scale_params=scale_params, shape=T2w_data.shape[1:],
                           verbose=True)
    stop = time.time()
    print("Scale took {} seconds".format(stop - start))
    scaled = nib.Nifti1Image(scaled.numpy()[0], scaled.affine.numpy())
    scaled.to_filename("data/examples/T2w/scaled_{}_{}_{}_T2w.nii.gz".format(*scale_params.tolist()))
    view_data(scaled, output_file="data/examples/T2w/scaled_{}_{}_{}_T2w_view.png".format(*scale_params.tolist()))
    plot_anat(scaled, title="Scaled T2w", cut_coords=coords,
              output_file="data/examples/T2w/scaled_{}_{}_{}_T2w.png".format(*scale_params.tolist()))

    # example of a monai scale
    start = time.time()
    scaled = monai_augment_image(image=T2w_data, translate_params=None, rotate_params=None,
                                 flip_params=None, shear_params=None, scale_params=scale_params)
    stop = time.time()
    print("Monai scale took {} seconds".format(stop - start))
    scaled = nib.Nifti1Image(scaled.numpy()[0], scaled.affine.numpy())
    scaled.to_filename("data/examples/T2w/monai_scaled_{}_{}_{}_T2w.nii.gz".format(*scale_params.tolist()))
    view_data(scaled, output_file="data/examples/T2w/monai_scaled_{}_{}_{}_T2w_view.png".format(*scale_params.tolist()))
    plot_anat(scaled, title="Monai Scaled T2w", cut_coords=coords,
              output_file="data/examples/T2w/monai_scaled_{}_{}_{}_T2w.png".format(*scale_params.tolist()))


if __name__ == "__main__":
    with torch.no_grad():
        main()
