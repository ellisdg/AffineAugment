import torch


def translate(x, y, z):
    return torch.tensor([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]])


def scale(x, y, z):
    return torch.tensor([[x, 0, 0, 0],
                         [0, y, 0, 0],
                         [0, 0, z, 0],
                         [0, 0, 0, 1]])


def rotate_x(theta):
    return torch.tensor([[1, 0, 0, 0],
                         [0, torch.cos(theta), -torch.sin(theta), 0],
                         [0, torch.sin(theta), torch.cos(theta), 0],
                         [0, 0, 0, 1]])


def rotate_y(theta):
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta), 0],
                         [0, 1, 0, 0],
                         [-torch.sin(theta), 0, torch.cos(theta), 0],
                         [0, 0, 0, 1]])


def rotate_z(theta):
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0, 0],
                         [torch.sin(theta), torch.cos(theta), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


def rotate_90_x(n=1):
    return rotate_x(torch.tensor(n*torch.pi/2))


def rotate_90_y(n=1):
    return rotate_y(torch.tensor(n*torch.pi/2))


def rotate_90_z(n=1):
    return rotate_z(torch.tensor(n*torch.pi/2))


def shear_x(y, z):
    return torch.tensor([[1, y, z, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


def shear_y(x, z):
    return torch.tensor([[1, 0, 0, 0],
                         [x, 1, z, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


def shear_z(x, y):
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [x, y, 1, 0],
                         [0, 0, 0, 1]])


def reflect_x():
    return scale(-1, 1, 1)


def reflect_y():
    return scale(1, -1, 1)


def reflect_z():
    return scale(1, 1, -1)

