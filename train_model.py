from monai.networks.nets import DynUNet
from monai.data import MetaTensor
from monai.utils import set_determinism
from torch.utils.data import DataLoader, Dataset
import glob
import os
import nibabel as nib
import torch
from augment import augment_image, monai_augment_image
import argparse


class T2wToT1wDataset(Dataset):
    def __init__(self, t2w_files, augment=None, normalize=True, augmentation_probability=0.5):
        self.augment = augment
        self.normalize = normalize
        self.augmentation_probability = augmentation_probability
        temp_t2w_files = t2w_files
        self.t1w_files = list()
        self.t2w_files = list()
        for t2w_file in temp_t2w_files:
            t1w_file = t2w_file.replace("T2w", "T1w")
            if os.path.exists(t1w_file) and os.path.exists(t2w_file):
                self.t2w_files.append(t2w_file)
                self.t1w_files.append(t1w_file)

    def __len__(self):
        return len(self.t2w_files)

    def __getitem__(self, idx):
        t2w_file = self.t2w_files[idx]
        t2w_image = nib.load(t2w_file)
        t2w_data = MetaTensor(torch.from_numpy(t2w_image.get_fdata()), affine=torch.from_numpy(t2w_image.affine))
        t1w_file = self.t1w_files[idx]
        t1w_image = nib.load(t1w_file)
        t1w_data = MetaTensor(torch.from_numpy(t1w_image.get_fdata()), affine=torch.from_numpy(t1w_image.affine))
        if self.normalize:
            t2w_data = (t2w_data - t2w_data.mean()) / torch.std(t2w_data)
            t1w_data = (t1w_data - t1w_data.mean()) / torch.std(t1w_data)
        if self.augment == "monai":
            t2w_data, t1w_data = self.monai_augment(t2w_data, t1w_data)
        elif self.augment == "mine":
            t2w_data, t1w_data = self.my_augment(t2w_data, t1w_data)
        return t2w_data, t1w_data

    def generate_augmentation_parameters(self):
        if torch.rand(1) < self.augmentation_probability:
            # generate translation parameters with mean 0 and std 10
            translate_params = torch.randn(3) * 10
        else:
            translate_params = None
        if torch.rand(1) < self.augmentation_probability:
            # generate rotation parameters with mean 0 and std 0.5
            rotate_params = torch.randn(3) * 0.5
        else:
            rotate_params = None
        if torch.rand(1) < self.augmentation_probability:
            # generate flip parameters
            flip_params = torch.rand(3) > 0.5
        else:
            flip_params = None
        if torch.rand(1) < self.augmentation_probability:
            # generate shear params with mean 0 and std 0.5
            shear_params = torch.randn(3) * 0.5
        else:
            shear_params = None
        if torch.rand(1) < self.augmentation_probability:
            # generate scale params with mean 1 and std 0.1
            scale_params = torch.randn(3) * 0.1 + 1
        else:
            scale_params = None
        return translate_params, rotate_params, flip_params, shear_params, scale_params

    def monai_augment(self, t2w_data, t1w_data):
        translate_params, rotate_params, flip_params, shear_params, scale_params = self.generate_augmentation_parameters()
        t2w_data = monai_augment_image(image=t2w_data, translate_params=translate_params, rotate_params=rotate_params,
                                       flip_params=flip_params, shear_params=shear_params, scale_params=scale_params)
        t1w_data = monai_augment_image(image=t1w_data, translate_params=translate_params, rotate_params=rotate_params,
                                       flip_params=flip_params, shear_params=shear_params, scale_params=scale_params)
        return t2w_data, t1w_data

    def my_augment(self, t2w_data, t1w_data):
        translate_params, rotate_params, flip_params, shear_params, scale_params = self.generate_augmentation_parameters()
        t2w_data = augment_image(image=t2w_data, translate_params=translate_params, rotate_params=rotate_params,
                                 flip_params=flip_params, shear_params=shear_params, scale_params=scale_params)
        t1w_data = augment_image(image=t1w_data, translate_params=translate_params, rotate_params=rotate_params,
                                 flip_params=flip_params, shear_params=shear_params, scale_params=scale_params)
        return t2w_data, t1w_data


def get_model():
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        filters=[4, 8, 16, 32, 64, 256],
        strides=[[1, 1, 1],
                 [2, 2, 2],
                 [2, 2, 2],
                 [2, 2, 2],
                 [2, 2, 2],
                 [2, 2, 2]],
        kernel_size=[[3, 3, 3],
                     [3, 3, 3],
                     [3, 3, 3],
                     [3, 3, 3],
                     [3, 3, 3],
                     [3, 3, 3]],
        upsample_kernel_size=[[2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2]])

    return model


def train_validation_split(filenames, validation_size=0.2, random_state=25):
    from numpy import random
    random.seed(random_state)
    random.shuffle(filenames)
    split_idx = int(len(filenames) * (1 - validation_size))
    train_filenames = filenames[:split_idx]
    val_filenames = filenames[split_idx:]
    return train_filenames, val_filenames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", type=str, default=None, help="augmentation method: (None, monai, mine)")
    parser.add_argument("--augmentation_probability", type=float, default=0.5, help="probability of augmentation")
    parser.add_argument("--epochs", type=int, default=250, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--model_filename", type=str, default="model.pt", help="filename to save model")
    return parser.parse_args()


def main():
    args = parse_args()
    # Set the seed for reproducibility
    set_determinism(seed=25)
    filenames = glob.glob("/work/aizenberg/dgellis/HCP/HCP_1200/*/T2w/T2w_acpc_dc_restore_brain.nii.gz")
    # split filenames into train and validation
    train_filenames, val_filenames = train_validation_split(filenames, validation_size=0.2, random_state=25)
    train_dataset = T2wToT1wDataset(train_filenames, augment=args.augment,
                                    augmentation_probability=args.augmentation_probability)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=30)
    val_dataset = T2wToT1wDataset(val_filenames)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=30)
    model = torch.nn.DataParallel(get_model()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss().cuda()
    for epoch in range(args.epochs):
        print("Epoch: {}".format(epoch))
        for i, (t2w_data, t1w_data) in enumerate(train_loader):
            t2w_data = t2w_data.cuda()
            t1w_data = t1w_data.cuda()
            optimizer.zero_grad()
            output = model(t2w_data)
            loss = loss_function(output, t1w_data)
            loss.backward()
            optimizer.step()
            print("Step {}/{}; Loss: {}".format(i, int(len(train_loader)/2), loss.item()))
        for i, (t2w_data, t1w_data) in enumerate(val_loader):
            t2w_data = t2w_data.cuda()
            t1w_data = t1w_data.cuda()
            output = model(t2w_data)
            loss = loss_function(output, t1w_data)
            print("Validation Step {}/{}; Loss: {}".format(i, int(len(val_loader)/2), loss.item()))
        torch.save(model.state_dict(), args.model_filename)


if __name__ == "__main__":
    main()
