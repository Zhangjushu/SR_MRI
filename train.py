import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from model.HDFA import HDFA
from datasets.get_h5 import TrainDataset, ValDatasetFromFolder
from utils import AverageMeter
import pytorch_ssim
from math import log10

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='')
parser.add_argument('--train_file', type=str, default='')
parser.add_argument('--eval_file', type=str, default='')
parser.add_argument('--outputs_dir', type=str, default='checkpoints')
parser.add_argument('--summary', type=str, default='logs')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--weights_file', type=str, default=None)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=48)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--weight_save_epoch', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--statistics', type=str, default='statistics')
parser.add_argument('--decay_iter', type=list, default=[100, 200])
parser.add_argument('--lr_decay_epoch', type=int, default=100)

if __name__ == '__main__':

    # 预处理
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    args.outputs_dir = os.path.join(args.outputs_dir, args.arch)
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    args.statistics = os.path.join(args.statistics, args.arch)
    if not os.path.exists(args.statistics):
        os.makedirs(args.statistics)

    args.summary = os.path.join(args.summary, args.arch)
    if not os.path.exists(args.summary):
        os.makedirs(args.summary)
    writer = SummaryWriter(log_dir=args.summary)

    # 加载数据
    train_dataset = TrainDataset(args.train_file, patch_size=args.patch_size, scale=args.scale)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = ValDatasetFromFolder(args.eval_file, args.scale)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # 加载模型
    model = HDFA().to(device)
    # 损失函数
    criterion = nn.L1Loss().to(device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 加载断点模型
    if args.resume:
        print("*** Load save model file ***")
        checkpoint = torch.load(args.weights_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        args.start_epoch = checkpoint['start_epoch']

    # 优化器学习率衰减
    ep_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.decay_iter)

    # 训练
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    results = {'loss': [], 'psnr': [], 'ssim': []}

    for epoch in range(args.start_epoch, args.num_epochs):

        # 训练
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.num_epochs))

            for data in train_dataloader:

                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        # 验证
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(eval_dataloader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] = valing_results['batch_sizes'] + batch_size

                lr = val_lr.to(device)
                hr = val_hr.to(device)

                sr = model(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] = valing_results['mse'] + batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] = valing_results['ssims'] + batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim'])
                )

        # 学习率衰减
        ep_lr_scheduler.step()
        print('current learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

        # tensorboard监测训练
        writer.add_scalar('Train/Loss', epoch_losses.avg, epoch + 1)
        writer.add_scalar('Valid/psnr', valing_results['psnr'], epoch + 1)
        writer.add_scalar('Valid/ssim', valing_results['ssim'], epoch + 1)

        # 保存训练模型
        if (epoch+1) % args.weight_save_epoch == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "start_epoch": epoch + 1
            }
            torch.save(checkpoint, os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch + 1)))

        # 记录最佳值
        if valing_results['psnr'] > best_psnr:
            best_epoch = epoch + 1
            best_psnr = valing_results['psnr']
            best_weights = copy.deepcopy(model.state_dict())

        # 保存训练记录
        results['loss'].append(epoch_losses.avg)
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        data_frame = pd.DataFrame(
            data={'Loss': results['loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(args.start_epoch, epoch + 1))
        data_frame.to_csv(args.statistics + '/x{}'.format(str(args.scale)) + '_train_results.csv',
                          index_label='Epoch')

    # 保存并打印性能最好的模型
    print('best epoch: {}, psnr: {:.6f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

    # 关闭tensorboard监控
    writer.close()