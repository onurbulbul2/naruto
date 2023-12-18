from sys import path

#path.append("../")
import time
import random
import argparse
import torch
import os
from copy import deepcopy
from fedlab.utils.functional import get_best_gpu
from fedlab.utils.logger import Logger
from fedlab.utils.serialization import SerializationTool
#from trainer import PerFedAvgTrainer
#from fine_tuner import LocalFineTuner
from utils import get_args, get_optimizer
from matplotlib import pyplot as plt
import argparse
from tqdm import trange

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

path.append("../..")
from PointNetGPD.model.dataset import *
from PointNetGPD.model.pointnet import PointNetCls

parser = argparse.ArgumentParser()
args = get_args(parser)

os.makedirs(args.model_path, exist_ok=True)

#確保了多個工作進程的隨機性是可重現的
np.random.seed(int(time.time()))
def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2 ** 31 - 1))

#過濾掉 batch 中值為 None 的樣本。
# lambda x: x is not None 這個 lambda 函數被用來檢查每個樣本是否為 None，然後 filter 函數將 batch 中不為 None 的樣本保留下來。
def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

grasp_points_num = 750
thresh_good = 0.6
thresh_bad = 0.6
point_channel = 3

def train(model, loader, epoch):
    log = SummaryWriter(os.path.join('./assets/log/', args.tag))
    optimizer = optim.Adam(model.parameters(), lr=args.local_lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    gradients = []
    # 保存當前模型的參數
    current_params = [param.data.clone() for param in model.parameters()]
    # 遍歷數據加載器，每次獲取一個 batch 的數據
    for batch_idx, (data, target) in enumerate(loader):
        if len(loader) == 0:
            print("Empty DataLoader. No data is provided.")
            break
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # 將梯度歸零，執行前向傳播，計算損失，執行反向傳播，然後根據優化器進行一步優化。
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # 計算準確率並記錄訓練損失
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()

        # 如果達到了指定的 log 間隔，則輸出當前的訓練損失
        if batch_idx % args.log_interval == 0:
            percentage = 100. * batch_idx * args.batch_size / len(loader.dataset)
            print(f'Train Epoch: {epoch} [{batch_idx * args.batch_size}/{len(loader.dataset)} ({percentage}%)]'
                  f'\tLoss: {loss.item()}\t{args.tag}')
            log.add_scalar('train_loss', loss.cpu().item(), batch_idx + epoch * len(loader))

        # 保存梯度（計算梯度和先前保存的模型參數之間的差異）
        for old_param, new_param in zip(current_params, model.parameters()):
            gradients.append(old_param - new_param.data)
    weight = torch.tensor(len(loader.sampler))
    return float(correct) / float(dataset_size), weight, gradients

def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    da = {}
    db = {}
    res = []
    ##遍歷測試數據，每次獲取一個 batch 的數據
    for data, target, obj_name in loader:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        ##進行模型推理，計算並累加測試損失。這裡使用了負的自然對數概率損失（F.nll_loss）
        output, _ = model(data)  # N*C
        test_loss += F.nll_loss(output, target, reduction='sum').cpu().item()
        # 計算正確預測的數量
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        # 記錄每個樣本的預測結果
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j[0], k))
    # 計算準確率和平均測試損失
    test_loss /= len(loader.dataset)
    acc = float(correct) / float(dataset_size)
    return acc, test_loss
def main():
    logger = Logger(log_name="Personalized FedAvg")
    '''
    device = torch.device("cpu")
    if torch.cuda.is_available() and args.cuda:
        device = get_best_gpu()
    global_model = PointNetCls(num_points=grasp_points_num, input_chann=point_channel, k=2).to(device)
    global_optimizer = get_optimizer(
        global_model, "sgd", dict(lr=args.server_lr, momentum=0.9)
    )
    criterion = torch.nn.CrossEntropyLoss()
    '''
    # seperate clients into training clients & test clients
    num_training_clients = int(0.8 * args.client_num_in_total)
    training_clients_id_list = range(num_training_clients)
    num_testing_clients = args.client_num_in_total - num_training_clients
    test_clients_id_list = range(num_testing_clients)
    '''
    print("num_training_clients = {}".format(num_training_clients))
    print("num_testing_clients = {}".format(num_testing_clients))
    print("training_clients_id_list = {}".format(training_clients_id_list))
    print("test_clients_id_list = {}".format(test_clients_id_list))
    '''
    is_resume = 0
    if args.load_model and args.load_epoch != -1:
        is_resume = 1

    if is_resume or args.mode == 'test':
        global_model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
        global_model.device_ids = [args.gpu]
        print('load model {}'.format(args.load_model))
    else:
        global_model = PointNetCls(num_points=grasp_points_num, input_chann=point_channel, k=2)
        global_optimizer = get_optimizer(
            global_model, "sgd", dict(lr=args.server_lr, momentum=0.9)
        )
        criterion = torch.nn.CrossEntropyLoss()

    if args.cuda:
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            model = global_model.cuda()
        else:
            device_id = [0, 1, 2, 3]
            torch.cuda.set_device(device_id[0])
            model = nn.DataParallel(global_model, device_ids=device_id).cuda()

    stats = dict(init=[], per=[])
    # FedAvg training
    for e in range(args.epochs):
        logger.info(f"FedAvg training epoch [{e}] ")
        selected_clients = random.sample(
            training_clients_id_list, args.client_num_per_round
        )
        all_client_weights = []
        all_client_gradients = []

        for client_id in selected_clients:
            logger.info(f"choose client = [{client_id}] ")
            args.tag = "client_{}".format(client_id)
            log = SummaryWriter(os.path.join('./assets/log/', args.tag))

            train_loader = torch.utils.data.DataLoader(
                PointGraspOneViewDataset(
                    grasp_points_num=grasp_points_num,
                    tag='train',
                    grasp_amount_per_file=6500,
                    thresh_good=thresh_good,
                    thresh_bad=thresh_bad,
                    client_id=client_id,
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
                    client_id=client_id,
                ),
                batch_size=args.batch_size,
                num_workers=20,
                pin_memory=True,
                shuffle=True,
                worker_init_fn=worker_init_fn,
                collate_fn=my_collate,
            )

            if args.mode == 'train':
                for epoch in range(is_resume * args.load_epoch,  args.inner_loops):
                    acc_train, weight, grads = train(global_model, train_loader, epoch)
                    print('Train done, acc={}'.format(acc_train))
                    acc, loss = test(global_model, test_loader)
                    print('Test done, acc={}, loss={}'.format(acc, loss))
                    # 使用 tensorboardX 的 SummaryWriter 將訓練和測試的準確率及損失記錄下來。
                    log.add_scalar('train_acc', acc_train, epoch)
                    log.add_scalar('test_acc', acc, epoch)
                    log.add_scalar('test_loss', loss, epoch)
                    # 每隔一定的 epoch 數（由 args.save_interval 決定），保存模型的狀態。
                    if epoch+1 % args.save_interval == 0 :
                        path = os.path.join(args.model_path, 'client' + '_{}_{}.model'.format(client_id, epoch))
                        torch.save(global_model, path)
                        print('Save model @ {}'.format(path))
                    if epoch+1 % args.inner_loops == 0 and epoch > 0:
                        print("save weight and grads every {} epochs ".format(epoch))
                        all_client_weights.append(weight)
                        all_client_gradients.append(grads)
            else:
                print('testing...')
                acc, loss = test(global_model, test_loader)
                print('Test done, acc={}, loss={}'.format(acc, loss))

        # FedAvg aggregation(using momentum SGD)
        global_optimizer.zero_grad()
        weights_sum = sum(all_client_weights)
        all_client_weights = [weight / weights_sum for weight in all_client_weights]
        for weight, grads in zip(all_client_weights, all_client_gradients):
            for param, grad in zip(global_model.parameters(), grads):
                if param.grad is None:
                    param.grad = torch.zeros(
                        param.size(), requires_grad=True, device=param.device
                    )
                param.grad.data.add_(grad.data * weight)
        global_optimizer.step()
'''
        if e % 20 == 0:
            selected_clients = random.sample(
                test_clients_id_list, args.client_num_per_round
            )
            init_acc = per_acc = 0
            for client_id in selected_clients:
                init_stats, per_stats = trainer.evaluate(
                    client_id, SerializationTool.serialize_model(global_model)
                )
                init_acc += init_stats[1]
                per_acc += per_stats[1]
            stats["init"].append(init_acc / args.client_num_per_round)
            stats["per"].append(per_acc / args.client_num_per_round)

    # Plot
    if os.path.isdir("./image") is False:
        os.mkdir("./image")
    plt.plot(stats["init"])
    plt.plot(stats["per"])
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.xticks(range(len(stats["init"])), range(0, len(stats["init"]) * 20, 20))
    plt.legend(
        [
            "E = {}  init acc".format(args.inner_loops),
            "E = {}  pers acc".format(args.inner_loops),
        ]
    )
    plt.savefig(
        "./image/E={} {} clients".format(args.inner_loops, args.client_num_per_round)
    )

    # Fine-tune
    if args.fine_tune:
        server_optimizer = get_optimizer(
            global_model, "sgd", dict(lr=args.fine_tune_server_lr)
        )
        fine_tuner = LocalFineTuner(
            deepcopy(global_model),
            optimizer_type="adam",
            optimizer_args=dict(lr=args.fine_tune_local_lr, betas=(0, 0.999)),
            criterion=criterion,
            epochs=args.fine_tune_inner_loops,
            batch_size=args.batch_size,
            cuda=args.cuda,
            logger=Logger(log_name="fine-tune"),
        )
        logger.info(
            "\033[1;33mFine-tune start(epoch={})\033[0m".format(
                args.fine_tune_outer_loops
            )
        )
        for e in range(args.fine_tune_outer_loops):
            logger.info("Fine-tune epoch [{}] start".format(e))
            serialized_model_param = SerializationTool.serialize_model(global_model)
            all_clients_gradients = []
            selected_clients = random.sample(
                training_clients_id_list, args.client_num_per_round
            )
            for client_id in selected_clients:
                # send model to clients and retrieve gradients
                grads = fine_tuner.train(client_id, serialized_model_param)
                all_clients_gradients.append(grads)

            # aggregate grads and update model
            server_optimizer.zero_grad()
            for grads in all_clients_gradients:
                for param, grad in zip(global_model.parameters(), grads):
                    if param.grad is None:
                        param.grad = torch.zeros(
                            param.size(), requires_grad=True, device=param.device
                        )
                    param.grad.data.add_(grad.data.to(param.device))
            for param in global_model.parameters():
                param.grad.data.div_(len(selected_clients))
            server_optimizer.step()
        logger.info("Fine-tune end")

    # Personalization and final Evaluation
    avg_init_loss = avg_init_acc = avg_per_loss = avg_per_acc = 0
    for _ in range(args.test_round):
        init_stats = []
        per_stats = []
        selected_clients = random.sample(
            test_clients_id_list, args.client_num_per_round
        )

        for client_id in selected_clients:
            init_, per_ = trainer.evaluate(
                client_id, SerializationTool.serialize_model(global_model)
            )
            init_stats.append(init_)
            per_stats.append(per_)

        init_loss = init_acc = per_loss = per_acc = 0
        for i in range(len(selected_clients)):
            init_loss += init_stats[i][0]
            init_acc += init_stats[i][1]
            per_loss += per_stats[i][0]
            per_acc += per_stats[i][1]

        avg_init_loss += init_loss / (args.client_num_per_round * args.test_round)
        avg_init_acc += init_acc / (args.client_num_per_round * args.test_round)
        avg_per_loss += per_loss / (args.client_num_per_round * args.test_round)
        avg_per_acc += per_acc / (args.client_num_per_round * args.test_round)

    print(
        "\033[1;33m-------------------------------- RESULTS --------------------------------\033[0m"
    )
    print(
        "\033[1;33minit loss: {:.4f}\ninit acc: {:.1f}%\nper loss: {:.4f}\nper acc: {:.1f}%\033[0m".format(
            avg_init_loss, (avg_init_acc * 100.0), avg_per_loss, (avg_per_acc * 100.0)
        )
    )
'''

if __name__ == "__main__":
    main()