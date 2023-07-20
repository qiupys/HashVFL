import argparse
import ast
import logging
import os
import time

from data.data import loadDataset, dataLoader
from utils.learn import train, test
from utils.model import CustomResNet, Server

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--seed', default=128, type=int, help='random seed')
parser.add_argument('--dataset_name', default='cifar10', type=str, help='dataset')
parser.add_argument('--encode_length', default=4, type=int, help='hash length')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--defense', default=False, type=ast.literal_eval, help='whether use defense strategy or not')
parser.add_argument('--type', default='resnet18', type=str, help='model architecture of the bottom model')
parser.add_argument('--num_layer', default=1, type=int, help='number of layers of the top model')
args = parser.parse_args()

# environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import torch
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

# setup logger
filename = "{}_{}_{}_{}_{}.txt".format(time.strftime('%Y%m%d%H%M%S'), args.dataset_name, args.defense, args.type,
                                       args.num_layer)
logger = setupLogger(filename)

# load models
logging.info('Preparing model')
modelbase = '/home/qpy/vhash/pretrained' if os.path.exists(
    '/home/qpy/vhash/pretrained') else 'C:\\Users\\Qiupys\\PycharmProjects\\vhash\\pretrained'
directory = "{}_{}_{}_{}".format(args.dataset_name, args.defense, args.type, args.num_layer)


def prepareModels(dataset_name, type, num_features, encode_length, defense, num_party, num_classes, num_layers,
                  device):
    workers = []
    if dataset_name in ['mnist', 'cifar10', 'cifar100', 'emotion']:
        in_channels = 1 if dataset_name in ['mnist'] else 3
        for _ in range(len(num_features)):
            workers.append(
                CustomResNet(type=type, in_channels=in_channels, encode_length=encode_length, defense=defense,
                             device=device).to(device))
    else:
        logging.info('Not supported datatype!')
        exit()
    server = Server(num_party=num_party, in_features=encode_length, num_classes=num_classes, num_layers=num_layers).to(
        device)
    return workers, server


if not os.path.exists(os.path.join(modelbase, directory)):
    os.makedirs(os.path.join(modelbase, directory))
    logging.info('Pretrained models do not exist! Begin training...')
    workers, server = prepareModels(dataset_name=dataset_name, type=args.type, num_features=num_features,
                                    encode_length=args.encode_length, defense=args.defense, num_party=args.num_party,
                                    num_classes=num_classes, num_layers=args.num_layer, device=device)
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
        train(epoch=epoch, workers=workers, server=server, data_loaders=train_data_loader, defense=args.defense, lr=lr,
              weight_decay=weight_decay, ortho_target=orthoTarget, device=device)
        test(epoch=epoch, workers=workers, server=server, data_loaders=test_data_loader, defense=args.defense,
             device=device)

    # save models
    for i, worker in enumerate(workers):
        torch.save(worker.state_dict(), os.path.join(modelbase, directory, 'worker_{}.pt'.format(i)))
    torch.save(server.state_dict(), os.path.join(modelbase, directory, 'server.pt'))
else:
    logging.info('Pretrained models have been saved.')
