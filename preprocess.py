"""
find the smallest viable image size for the dataset to the nearest multiple of 32 and crop all images to that size.
"""
import glob
import nibabel as nib
import numpy as np
from nilearn.image import crop_img
from nilearn.image.image import  _crop_img_to


def main():
    filenames = glob.glob("/work/aizenberg/dgellis/HCP/HCP_1200/*/T1w/T*w_acpc_dc_restore_brain.nii.gz")
    mask = None
    for filename in filenames:
        image = nib.load(filename)
        if mask is None:
            mask = image.get_fdata() > 0
        else:
            mask = mask & (image.get_fdata() > 0)

    mask = nib.Nifti1Image(np.asarray(mask, np.int16), image.affine)
    print("Original image shape:", mask.shape)

    # Find the cropped shape
    cropped_mask, slices = crop_img(mask, pad=False, return_offset=True)
    print(slices)
    cropped_shape = cropped_mask.shape
    print("Cropped image shape:", cropped_shape)

    # crop to nearest multiple of 32
    final_shape = tuple([int(np.ceil(s / 32) * 32) for s in cropped_shape])
    final_slices = list()
    for s, shape, og_shape in zip(slices, final_shape, mask.shape):
        s_start = s.start
        s_end = s.stop
        while s_end - s_start < shape:
            if s_end < og_shape:
                s_end = s_end + 1
            if s_end - s_start < shape and s_start > 0:
                s_start = s_start - 1
        final_slices.append(slice(s_start, s_end, None))

    print(final_slices)
    print([s.stop - s.start for s in final_slices], final_shape)

    for filename in filenames:
        image = nib.load(filename)
        cropped_image = _crop_img_to(image, final_slices, copy=False)
        cropped_image.to_filename(filename.replace(".nii.gz", "_cropped.nii.gz"))
        print(cropped_image.shape)


if __name__ == "__main__":
    main()
