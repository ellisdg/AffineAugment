import torch
from augment import augment_image, monai_augment_image
import nibabel as nib
from monai.data import MetaTensor


def main():
    t1w_image = nib.load("data/T1w.nii.gz")
    t1w_data = MetaTensor(torch.from_numpy(t1w_image.get_fdata()[None]).to(dtype=torch.float32),
                          affine=torch.from_numpy(t1w_image.affine).to(dtype=torch.float32))
    # example of a translation
    translate_params = torch.tensor([10, 10, 10])
    translated = augment_image(image=t1w_data.detach().clone(), translate_params=translate_params, rotate_params=None,
                               flip_params=None, shear_params=None, scale_params=None, shape=t1w_data.shape[1:])
    translated = nib.Nifti1Image(translated.numpy()[0], translated.affine.numpy())
    translated.to_filename("data/examples/translated_{}_T1w.nii.gz".format("_".join(
        [str(x) for x in translate_params.tolist()])))

    # example of a monai translation
    translated = monai_augment_image(image=t1w_data.detach().clone(), translate_params=translate_params,
                                     rotate_params=None, flip_params=None, shear_params=None, scale_params=None,
                                     keep_size=False)
    translated = nib.Nifti1Image(translated.numpy()[0], translated.affine.numpy())
    translated.to_filename("data/examples/monai_translated_{}_T1w.nii.gz".format("_".join(
        [str(x) for x in translate_params.tolist()])))

    # example of a rotation
    rotated = augment_image(image=t1w_data, translate_params=None, rotate_params=torch.tensor([0.5, 0.5, 0.5]),
                            flip_params=None, shear_params=None, scale_params=None, shape=t1w_data.shape[1:])
    rotated = nib.Nifti1Image(rotated.numpy()[0], rotated.affine.numpy())
    rotated.to_filename("data/examples/rotated_05_05_05_T1w.nii.gz")


if __name__ == "__main__":
    with torch.no_grad():
        main()
