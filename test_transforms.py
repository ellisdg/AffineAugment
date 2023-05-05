from monai.transforms import Rotate, Zoom, Flip, Crop, Affine
from monai.data import MetaTensor
import torch
from affine import augment
import unittest


class TestTransforms(unittest.TestCase):
    def setUp(self):
        # Create a test image to augment
        affine = torch.rand(4, 4)
        affine[3, :] = torch.tensor([0, 0, 0, 1])
        self.image = MetaTensor(x=torch.rand(1, 100, 100, 100), affine=affine)

    def test_rotate_x(self):
        angle = torch.tensor([0.5, 0, 0])
        monai_rotate_x = Rotate(angle=angle)(self.image)
        affine_rotate_x = augment(self.image.affine, rotate_params=angle, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_rotate_x.affine, affine_rotate_x))

    def test_rotate_y(self):
        angle = torch.tensor([0, 0.25, 0])
        monai_rotate_y = Rotate(angle=angle)(self.image)
        affine_rotate_y = augment(self.image.affine, rotate_params=angle, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_rotate_y.affine, affine_rotate_y))

    def test_rotate_z(self):
        angle = torch.tensor([0, 0, -0.5])
        monai_rotate_z = Rotate(angle=angle)(self.image)
        affine_rotate_z = augment(self.image.affine, rotate_params=angle, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_rotate_z.affine, affine_rotate_z))

    def test_rotate(self):
        angle = torch.tensor([0.5, 0.25, -0.5])
        monai_rotate = Rotate(angle=angle)(self.image)
        affine_rotate = augment(self.image.affine, rotate_params=angle, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_rotate.affine, affine_rotate))

    def test_translate(self):
        translation = torch.tensor([-10, 20, 30])
        monai_translate, _ = Affine(translate_params=translation.tolist())(self.image.detach().clone())
        # I have to use detach().clone() because the Affine transform modifies the image in place
        # (as well as returning an image, which is slightly confusing and not how the other transforms work)
        affine_translate = augment(self.image.affine, translate_params=translation)
        self.assertTrue(torch.allclose(monai_translate.affine, affine_translate))

    def test_zoom(self):
        zoom_params = torch.tensor([0.5, 1.5, 1.5])

        monai_zoom = Zoom(zoom=zoom_params, keep_size=False)(self.image)

        # The scaling parameters are the inverse of the zoom parameters
        affine_zoom = augment(self.image.affine, scale_params=1/zoom_params)

        self.assertTrue(torch.allclose(monai_zoom.affine, affine_zoom))

    def test_flip_x(self):
        flip_params = torch.tensor([True, False, False])
        monai_flip = Flip(0)(self.image)
        affine_flip = augment(self.image.affine, flip_params=flip_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_flip.affine, affine_flip))

    def test_flip_y(self):
        flip_params = torch.tensor([False, True, False])
        monai_flip = Flip(1)(self.image)
        affine_flip = augment(self.image.affine, flip_params=flip_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_flip.affine, affine_flip))

    def test_flip_z(self):
        flip_params = torch.tensor([False, False, True])
        monai_flip = Flip(2)(self.image)
        affine_flip = augment(self.image.affine, flip_params=flip_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_flip.affine, affine_flip))

    def test_flip_all(self):
        flip_params = torch.tensor([True, True, True])
        monai_flip = Flip()(self.image)
        affine_flip = augment(self.image.affine, flip_params=flip_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_flip.affine, affine_flip))

