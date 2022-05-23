import argparse
import torch
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image
import os
from tqdm import tqdm
import numpy as np
from model.HDFA import HDFA
from utils import denormalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='')
    parser.add_argument('--weights_file', type=str, default='')
    parser.add_argument('--image_file', type=str, default='')
    parser.add_argument('--data_name', type=str, default='')
    parser.add_argument('--outputs_dir', type=str, default='results')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir,   
                                    args.arch,
                                    args.data_name,
                                    f'x{args.scale}'
                                    )
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = HDFA().to(device)
    model.load_state_dict(torch.load(args.weights_file)['model_state_dict'])
    # model.load_state_dict(torch.load(args.weights_file))

    model.eval()

    images_name = [x for x in os.listdir(args.image_file)]
    for image_name in tqdm(images_name, desc='model test'):
        image = pil_image.open(args.image_file + image_name).convert('RGB')

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        lr.save(os.path.join(args.outputs_dir, '{}.png'.format(image_name.split('.')[0], args.scale)))

        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        lr = torch.from_numpy(lr).to(device)
        hr = torch.from_numpy(hr).to(device)

        with torch.no_grad():
            preds = model(lr).squeeze(0)

        output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
        output.save(os.path.join(args.outputs_dir, '{}.png'.format(image_name.split('.')[0])))
