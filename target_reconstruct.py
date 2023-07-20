import argparse
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.data import loadDataset, dataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--seed', default=128, type=int, help='random seed')
parser.add_argument('--dataset_name', default='mnist', type=str, help='dataset')
parser.add_argument('--encode_length', default=4, type=int, help='hash length')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--defense', default=True, type=bool, help='whether use defense strategy or not')
parser.add_argument('--target', default=0, type=int, help='target class for reconstruction')
parser.add_argument('--epochs', default=30000, type=int, help='iteration epochs')
args = parser.parse_args()

# environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import torch
from utils.function import setupSeed, feature_distribute, setupLogger, loadModels, loadOrthogonalTargets

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
feature_ratios = [0.5, 0.5]
assert len(feature_ratios) == args.num_party
num_features = feature_distribute(input_size, feature_ratios) if input_size > 0 else [max_length for _ in
                                                                                      range(args.num_party)]

# setup logger
filename = "reconstruct_{}_{}_{}_{}_{}.txt".format(time.strftime('%Y%m%d%H%M%S'), args.dataset_name,
                                                   args.encode_length, args.num_party, args.target)
logger = setupLogger(filename, False)

# load models
logging.info('Preparing model')
modelbase = '/home/qpy/vhash/pretrained' if os.path.exists(
    '/home/qpy/vhash/pretrained') else 'C:\\Users\\Qiupys\\PycharmProjects\\vhash\\pretrained'
directory = "{}_{}_{}_{}".format(args.dataset_name, args.encode_length, args.num_party, args.defense)

channel = 1 if dataset_name == 'mnist' else 3


# computes total variation for an image
def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def l2loss(x):
    return (x ** 2).mean()


if os.path.exists(os.path.join(modelbase, directory)):
    logging.info('Loading data')
    batch_size = config.getint(dataset_name, 'batch_size')
    train_datasets, test_datasets = loadDataset(dataset_name, args.num_party, num_features)

    workers, server = loadModels(root=os.path.join(modelbase, directory), dataset_name=dataset_name,
                                 num_features=num_features, encode_length=args.encode_length,
                                 defense=args.defense, epsilon=0, num_party=args.num_party,
                                 num_classes=num_classes, num_layers=1, device=device)

    # count = [0 for _ in range(10)]
    logging.info("Reconstruct Data...")
    real, y = test_datasets[0][1]
    workers[0].eval()
    real_embeds, real_codes = workers[0](real.unsqueeze(0).to(device))

    fake = torch.empty([1, channel, input_size, int(input_size / 2)]).fill_(0.5).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([fake], lr=1e-3, amsgrad=True)

    lambda_tv = 1
    lambda_l2 = 0.1
    for iter in tqdm(range(args.epochs)):
        workers[0].eval()
        optimizer.zero_grad()
        embeds, codes = workers[0](fake)
        mse_loss = torch.nn.MSELoss()(embeds, real_embeds)
        tv_loss = TV(fake)
        l2_loss = l2loss(fake)
        loss = mse_loss + lambda_tv * tv_loss + lambda_l2 * l2_loss
        loss.backward(retain_graph=True)
        optimizer.step()

    plt.subplot(1, 2, 1)
    real_img = np.transpose(real.numpy(), (1, 2, 0))
    plt.imshow(real_img)
    # np.save(os.path.join('C:\\Users\\Qiupys\\PycharmProjects\\vhash\\reconstruct',
    #                      '{}_{}_{}_ground_truth'.format(dataset_name, args.encode_length, y)), real_img)
    plt.subplot(1, 2, 2)
    fake_img = np.transpose(fake.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(fake_img)
    # np.save(os.path.join('C:\\Users\\Qiupys\\PycharmProjects\\vhash\\reconstruct',
    #                      '{}_{}_{}_no_defense'.format(dataset_name, args.encode_length, y)), fake_img)
    plt.show()
    # plt.imshow(value[0])
    # plt.show()


else:
    logging.info('Pretrained models do not exist! Please train first.')
