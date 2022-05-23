import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from torchvision.transforms import transforms


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    image_list = sorted(glob.glob('{}/*'.format(args.images_dir)))
    patch_idx = 0

    for i, image_path in enumerate(image_list):
        hr = pil_image.open(image_path).convert('RGB')

        ####
        # for hr in transforms.FiveCrop(size=(hr.height // 2, hr.width // 2))(hr):
        hr = hr.resize(((hr.width // args.scale) * args.scale, (hr.height // args.scale) * args.scale), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

        hr = np.array(hr)
        lr = np.array(lr)

        lr_group.create_dataset(str(patch_idx), data=lr)
        hr_group.create_dataset(str(patch_idx), data=hr)

        patch_idx += 1
        ####

        print(i, patch_idx, image_path)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    train(args)