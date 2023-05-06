from monai.transforms import Rotate, Zoom, Flip, Affine
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
        self.assertTrue(torch.allclose(monai_rotate_x.affine, affine_rotate_x, atol=1e-5))

    def test_rotate_y(self):
        angle = torch.tensor([0, 0.25, 0])
        monai_rotate_y = Rotate(angle=angle)(self.image)
        affine_rotate_y = augment(self.image.affine, rotate_params=angle, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_rotate_y.affine, affine_rotate_y, atol=1e-5))

    def test_rotate_z(self):
        angle = torch.tensor([0, 0, -0.5])
        monai_rotate_z = Rotate(angle=angle)(self.image)
        affine_rotate_z = augment(self.image.affine, rotate_params=angle, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_rotate_z.affine, affine_rotate_z, atol=1e-5))

    def test_rotate(self):
        angle = torch.tensor([0.5, 0.25, -0.5])
        monai_rotate = Rotate(angle=angle)(self.image)
        affine_rotate = augment(self.image.affine, rotate_params=angle, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_rotate.affine, affine_rotate, atol=1e-5))

    def test_translate(self):
        translation = torch.tensor([-10, 20, 30])
        monai_translate, _ = Affine(translate_params=translation.tolist())(self.image.detach().clone())
        # I have to use detach().clone() because the Affine transform modifies the image in place
        # (as well as returning an image, which is slightly confusing and not how the other transforms work)
        affine_translate = augment(self.image.affine, translate_params=translation)
        self.assertTrue(torch.allclose(monai_translate.affine, affine_translate, atol=1e-5))

    def test_zoom(self):
        zoom_params = torch.tensor([0.5, 1.5, 1.5])

        monai_zoom = Zoom(zoom=zoom_params, keep_size=False)(self.image)

        # The scaling parameters are the inverse of the zoom parameters
        affine_zoom = augment(self.image.affine, scale_params=1/zoom_params)

        self.assertTrue(torch.allclose(monai_zoom.affine, affine_zoom, atol=1e-5))

    def test_simple_flip(self):
        # Make sure that a 1x1x1 image with origin 0,0,0 will still have its origin at 0,0,0 after flipping
        flip_params = torch.tensor([True, True, True])
        image = MetaTensor(torch.rand(1, 1, 1, 1), affine=torch.eye(4))
        monai_flip = Flip()(image)
        assert torch.all(monai_flip.affine[:3, 3] == 0)
        affine_flip = augment(image.affine, flip_params=flip_params, shape=image.shape[1:])
        assert torch.all(affine_flip[:3, 3] == 0)

    def test_flip_x(self):
        flip_params = torch.tensor([True, False, False])
        monai_flip = Flip(0)(self.image)
        affine_flip = augment(self.image.affine, flip_params=flip_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_flip.affine, affine_flip, atol=1e-5))

    def test_flip_y(self):
        flip_params = torch.tensor([False, True, False])
        monai_flip = Flip(1)(self.image)
        affine_flip = augment(self.image.affine, flip_params=flip_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_flip.affine, affine_flip, atol=1e-5))

    def test_flip_z(self):
        flip_params = torch.tensor([False, False, True])
        monai_flip = Flip(2)(self.image)
        affine_flip = augment(self.image.affine, flip_params=flip_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_flip.affine, affine_flip, atol=1e-5))

    def test_flip_all(self):
        flip_params = torch.tensor([True, True, True])
        monai_flip = Flip()(self.image)
        affine_flip = augment(self.image.affine, flip_params=flip_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_flip.affine, affine_flip, atol=1e-5))

    def test_shear_x(self):
        shear_params = torch.rand(6)
        # Again, I need to clone the image because the Affine transform modifies the image in place
        monai_shear, _ = Affine(shear_params=shear_params.tolist())(self.image.detach().clone())
        affine_shear = augment(self.image.affine, shear_params=shear_params, shape=self.image.shape[1:])
        self.assertTrue(torch.allclose(monai_shear.affine, affine_shear, atol=1e-5))

