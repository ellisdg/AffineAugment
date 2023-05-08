import unittest
from monai.data import MetaTensor
import torch
from augment import augment_image, monai_augment_image


class TestAugment(unittest.TestCase):
    def setUp(self):
        self.image = MetaTensor(x=torch.rand(1, 100, 100, 100), affine=torch.eye(4))

    def test_augment(self):
        translate_params = torch.tensor([10, 20, -30])
        rotate_params = torch.tensor([0.5, 0.25, 0.5])
        flip_params = torch.tensor([False, True, True])
        shear_params = torch.tensor([0.9, 1.1, 0.8, 0.9, 1.1, 0.8])
        scale_params = torch.tensor([0.5, 2, 0.5])

        my_image = augment_image(self.image.detach().clone(),
                                 translate_params=translate_params,
                                 rotate_params=rotate_params,
                                 flip_params=flip_params,
                                 shear_params=shear_params,
                                 shape=self.image.shape[1:],
                                 scale_params=scale_params)
        monai_image = monai_augment_image(self.image.detach().clone(),
                                          translate_params=translate_params,
                                          rotate_params=rotate_params,
                                          flip_params=flip_params,
                                          shear_params=shear_params,
                                          scale_params=scale_params)

        self.assertTrue(torch.allclose(my_image.affine, monai_image.affine, atol=1e-5))