import argparse
import logging
import os
import time

from data.data import loadDataset, dataLoader
from utils.learn import adjust_learning_rate
from utils.model_without_bn import MLP, MyResNet, Server, BertBaseModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--seed', default=64, type=int, help='random seed')
parser.add_argument('--dataset_name', default='criteo', type=str, help='dataset')
parser.add_argument('--encode_length', default=4, type=int, help='hash length')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--method', default='ours', type=str, help='loss function design')
args = parser.parse_args()

# environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import torch
import torch.nn.functional as F
from utils.function import setupSeed, feature_distribute, generateOrthogonalTargets, setupLogger, generateFeatureRatios

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setupSeed(args.seed)

# read args and configures
from configparser import ConfigParser

config = ConfigParser()
config.read('./config/datasets.config', encoding='UTF-8')
dataset_name = args.dataset_name
input_size = config.getint(dataset_name, 'input_size')
max_length = config.getint(dataset_name, 'max_length') if dataset_name in ['imdb'] else None
num_classes = config.getint(dataset_name, 'num_classes')
feature_ratios = generateFeatureRatios(args.num_party)
assert len(feature_ratios) == args.num_party
num_features = feature_distribute(input_size, feature_ratios) if input_size > 0 else [max_length for _ in
                                                                                      range(args.num_party)]
defense = True if args.method in ['ours', 'greedy'] else False
epsilon = 0

# setup logger
filename = "{}_{}_{}_{}.txt".format(time.strftime('%Y%m%d%H%M%S'), args.dataset_name, args.encode_length, args.method)
logger = setupLogger(filename)

# load models
logging.info('Preparing model')
modelbase = '/home/qpy/vhash/pretrained' if os.path.exists(
    '/home/qpy/vhash/pretrained') else 'C:\\Users\\Qiupys\\PycharmProjects\\vhash\\pretrained'
directory = "{}_{}_{}".format(args.dataset_name, args.encode_length, args.method)


def prepareModels(dataset_name, num_features, encode_length, defense, epsilon, num_party, num_classes, num_layers,
                  device):
    workers = []
    if dataset_name in ['bank', 'criteo']:
        for num_feature in num_features:
            workers.append(
                MLP(in_features=num_feature, out_features=encode_length, defense=defense, epsilon=epsilon,
                    device=device, num_layers=num_layers).to(device))
    elif dataset_name in ['imdb']:
        for _ in range(len(num_features)):
            workers.append(BertBaseModel(encode_length, defense, epsilon, device).to(device))
    elif dataset_name in ['mnist', 'cifar10', 'cifar100', 'emotion']:
        in_channels = 1 if dataset_name in ['mnist'] else 3
        for _ in range(len(num_features)):
            workers.append(
                MyResNet(in_channels=in_channels, encode_length=encode_length, defense=defense, epsilon=epsilon,
                         device=device).to(device))
    else:
        logging.info('Not supported datatype!')
        exit()
    server = Server(num_party=num_party, in_features=encode_length, num_classes=num_classes, num_layers=1).to(device)
    return workers, server


def train(epoch, method, workers, server, data_loaders, lr, weight_decay, ortho_target, device):
    optimizers = [torch.optim.Adam(server.parameters(), lr=lr, weight_decay=weight_decay)]
    for worker in workers:
        if isinstance(worker, BertBaseModel):
            optimizers.append(torch.optim.Adam(worker.linear.parameters(), lr=lr, weight_decay=weight_decay))
        else:
            optimizers.append(torch.optim.Adam(worker.parameters(), lr=lr, weight_decay=weight_decay))
    adjust_learning_rate(lr, optimizers, epoch)
    part_data = [enumerate(data_loader) for data_loader in data_loaders]
    train_loss = 0
    correct = 0
    while True:
        for worker in workers:
            worker.train()
        server.train()
        try:
            part_X = []
            batch_idx, (X, y) = next(part_data[0])
            part_X.append(X.to(device))
            y = y.to(device)
            for pd in part_data[1:]:
                _, (X, _) = next(pd)
                part_X.append(X.to(device))
            embeds, codes = [], []
            for i in range(len(workers)):
                if isinstance(workers[i], BertBaseModel):
                    output, hashcode = workers[i](part_X[i], attention_mask=(part_X[i] > 0).to(device))
                    embeds.append(output)
                    codes.append(hashcode)
                else:
                    output, hashcode = workers[i](part_X[i])
                    embeds.append(output)
                    codes.append(hashcode)

            for optimizer in optimizers:
                optimizer.zero_grad()

            if method == 'ours':
                # classification loss
                preds = server(codes)
                cls_loss = F.nll_loss(preds, y)

                # cosine similarity loss
                cos_loss = 0
                num_classes = len(ortho_target)
                y_one_hot = F.one_hot(y, num_classes).type(torch.FloatTensor).to(device)
                target = torch.matmul(y_one_hot, ortho_target.to(device))
                for code in codes:
                    cos_loss -= torch.trace(torch.matmul(code, target.t())) / (len(code) * len(code[0]))
                cos_loss /= len(codes)

                loss = cls_loss + cos_loss
            elif method == 'greedy':
                # classification loss
                preds = server(codes)
                cls_loss = F.nll_loss(preds, y)

                # p-norm loss
                norm_loss = 0
                for i in range(len(embeds)):
                    norm_loss += torch.mean(torch.abs(torch.norm(embeds[i] - codes[i], p=3)))
                norm_loss /= len(embeds)

                loss = cls_loss + norm_loss
            else:
                # classification loss
                preds = server(codes)
                cls_loss = F.nll_loss(preds, y)
                loss = cls_loss
            train_loss = loss
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            correct += preds.max(1)[1].eq(y).sum().item()
        except StopIteration:
            break

    num_data = len(data_loaders[0].dataset)
    acc = correct / num_data
    train_loss = train_loss / num_data
    log = 'Epoch: {}, training Loss: {:.6f}, Acc:{:.4f}'
    logging.info(log.format(epoch, train_loss, acc))


def test(epoch, workers, server, data_loaders, defense, device):
    with torch.no_grad():
        part_data = [enumerate(loader) for loader in data_loaders]
        for worker in workers:
            worker.eval()
        server.eval()
        loss = 0
        correct = 0
        while True:
            try:
                part_X = []
                batch_idx, (X, y) = next(part_data[0])
                part_X.append(X.to(device))
                y = y.to(device)
                for pd in part_data[1:]:
                    _, (X, _) = next(pd)
                    part_X.append(X.to(device))
                embeds, codes = [], []
                for i in range(len(workers)):
                    if isinstance(workers[i], BertBaseModel):
                        output, hashcode = workers[i](part_X[i], attention_mask=(part_X[i] > 0).to(device))
                        embeds.append(output)
                        codes.append(hashcode)
                    else:
                        output, hashcode = workers[i](part_X[i])
                        embeds.append(output)
                        codes.append(hashcode)
                if defense:
                    preds = server(codes)
                else:
                    preds = server(embeds)
                loss += F.nll_loss(preds, y).item()
                correct += preds.max(1)[1].eq(y).sum().item()
            except StopIteration:
                break

        num_data = len(data_loaders[0].dataset)
        acc = correct / num_data
        loss = loss / num_data
        log = 'Epoch: {}, test Loss: {:.6f}, Acc:{:.4f}'
        logging.info(log.format(epoch, loss, acc))


if not os.path.exists(os.path.join(modelbase, directory)):
    os.makedirs(os.path.join(modelbase, directory))
    logging.info('Pretrained models do not exist! Begin training...')
    workers, server = prepareModels(dataset_name=dataset_name, num_features=num_features,
                                    encode_length=args.encode_length, defense=defense, epsilon=epsilon,
                                    num_party=args.num_party, num_classes=num_classes, num_layers=1, device=device)
    orthoTarget = generateOrthogonalTargets(num_classes, args.encode_length)
    torch.save(orthoTarget, os.path.join(modelbase, directory, 'orthTargets.pt'))

    # load data
    logging.info('Loading data')
    batch_size = config.getint(dataset_name, 'batch_size')
    train_datasets, test_datasets = loadDataset(dataset_name, args.num_party, num_features)
    train_data_loader, test_data_loader = dataLoader(train_datasets, batch_size), dataLoader(test_datasets, batch_size)

    # normal training
    lr, weight_decay = config.getfloat(dataset_name, 'lr'), config.getfloat(dataset_name, 'weight_decay')
    logging.info('Training...')
    for epoch in range(config.getint(dataset_name, 'epoch')):
        train(epoch=epoch, method=args.method, workers=workers, server=server, data_loaders=train_data_loader, lr=lr,
              weight_decay=weight_decay, ortho_target=orthoTarget, device=device)
        test(epoch=epoch, workers=workers, server=server, data_loaders=test_data_loader, defense=defense,
             device=device)

    # save models
    for i, worker in enumerate(workers):
        torch.save(worker.state_dict(), os.path.join(modelbase, directory, 'worker_{}.pt'.format(i)))
    torch.save(server.state_dict(), os.path.join(modelbase, directory, 'server.pt'))
else:
    logging.info('Pretrained models have been saved.')
