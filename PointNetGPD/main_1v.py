import argparse
import os
import time
import pickle

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from model.dataset import *
from model.pointnet import PointNetCls, DualPointNetCls

parser = argparse.ArgumentParser(description='pointnetGPD')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--load-model', type=str, default='')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                    help='pre-trained model path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=1)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False
if args.cuda:
    torch.cuda.manual_seed(1)

os.makedirs(args.model_path, exist_ok=True)
logger = SummaryWriter(os.path.join('./assets/log/', args.tag))

#確保了多個工作進程的隨機性是可重現的
np.random.seed(int(time.time()))
def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2 ** 31 - 1))

#過濾掉 batch 中值為 None 的樣本。
# lambda x: x is not None 這個 lambda 函數被用來檢查每個樣本是否為 None，
# 然後 filter 函數將 batch 中不為 None 的樣本保留下來。
def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


grasp_points_num = 750
thresh_good = 0.6
thresh_bad = 0.6
point_channel = 3


def train(model, loader, epoch):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    #遍歷數據加載器，每次獲取一個 batch 的數據
    for batch_idx, (data, target) in enumerate(loader):
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #將梯度歸零，執行前向傳播，計算損失，執行反向傳播，然後根據優化器進行一步優化。
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #計算準確率並記錄訓練損失
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        #如果達到了指定的 log 間隔，則輸出當前的訓練損失
        if batch_idx % args.log_interval == 0:
            percentage = 100. * batch_idx * args.batch_size / len(loader.dataset)
            print(f'Train Epoch: {epoch} [{batch_idx * args.batch_size}/{len(loader.dataset)} ({percentage}%)]'
                  f'\tLoss: {loss.item()}\t{args.tag}')
            logger.add_scalar('train_loss', loss.cpu().item(), batch_idx + epoch * len(loader))
    return float(correct) / float(dataset_size)


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    da = {}
    db = {}
    res = []
    #遍歷測試數據，每次獲取一個 batch 的數據
    for data, target, obj_name in loader:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #進行模型推理，計算並累加測試損失。這裡使用了負的自然對數概率損失（F.nll_loss）
        output, _ = model(data)  # N*C
        test_loss += F.nll_loss(output, target, reduction='sum').cpu().item()
        #計算正確預測的數量
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        #記錄每個樣本的預測結果
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j[0], k))
    #計算準確率和平均測試損失
    test_loss /= len(loader.dataset)
    acc = float(correct) / float(dataset_size)
    return acc, test_loss


def main():
    train_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            tag='train',
            grasp_amount_per_file=6500,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
        ),
        batch_size=args.batch_size,
        num_workers=20,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )

    test_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            tag='test',
            grasp_amount_per_file=500,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
            with_obj=True,
        ),
        batch_size=args.batch_size,
        num_workers=20,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )

    #如果 is_resume 為 1，則表示應該從之前保存的模型和指定的 epoch 處繼續訓練，而不是從頭開始。
    is_resume = 0
    if args.load_model and args.load_epoch != -1:
        is_resume = 1

    if is_resume or args.mode == 'test':
        model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
        model.device_ids = [args.gpu]
        print('load model {}'.format(args.load_model))
    else:
        model = PointNetCls(num_points=grasp_points_num, input_chann=point_channel, k=2)

    #看用幾個GPU
    if args.cuda:
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            model = model.cuda()
        else:
            device_id = [0, 1, 2, 3]
            torch.cuda.set_device(device_id[0])
            model = nn.DataParallel(model, device_ids=device_id).cuda()

    if args.mode == 'train':
        for epoch in range(is_resume * args.load_epoch, args.epoch):
            acc_train = train(model, train_loader, epoch)
            print('Train done, acc={}'.format(acc_train))
            acc, loss = test(model, test_loader)
            print('Test done, acc={}, loss={}'.format(acc, loss))
            #使用 tensorboardX 的 SummaryWriter 將訓練和測試的準確率及損失記錄下來。
            logger.add_scalar('train_acc', acc_train, epoch)
            logger.add_scalar('test_acc', acc, epoch)
            logger.add_scalar('test_loss', loss, epoch)
            #每隔一定的 epoch 數（由 args.save_interval 決定），保存模型的狀態。
            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc, loss = test(model, test_loader)
        print('Test done, acc={}, loss={}'.format(acc, loss))


if __name__ == "__main__":
    main()
