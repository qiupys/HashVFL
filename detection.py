import argparse
import logging
import math
import os
import time
import itertools

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--seed', default=128, type=int, help='random seed')
parser.add_argument('--dataset_name', default='mnist', type=str, help='dataset')
parser.add_argument('--encode_length', default=16, type=int, help='hash length')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--defense', default=True, type=bool, help='whether use defense strategy or not')
parser.add_argument('--epsilon', default=0, type=int, help='epsilon-DP guarantee')
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
filename = "{}_{}_{}_{}_{}_{}_detection.txt".format(time.strftime('%Y%m%d%H%M%S'), args.dataset_name,
                                                    args.encode_length,
                                                    args.num_party, args.defense, args.epsilon)
logger = setupLogger(filename)

# load models
logging.info('Preparing model')
modelbase = '/home/qpy/vhash/pretrained' if os.path.exists(
    '/home/qpy/vhash/pretrained') else 'C:\\Users\\Qiupys\\PycharmProjects\\vhash\\pretrained'
directory = "{}_{}_{}_{}_{}".format(args.dataset_name, args.encode_length, args.num_party, args.defense, args.epsilon)

if os.path.exists(os.path.join(modelbase, directory)):
    workers, server = loadModels(root=os.path.join(modelbase, directory), dataset_name=dataset_name,
                                 num_features=num_features, encode_length=args.encode_length,
                                 defense=args.defense, epsilon=args.epsilon, num_party=args.num_party,
                                 num_classes=num_classes, num_layers=1, device=device)
    orthoTarget = loadOrthogonalTargets(filename=os.path.join(modelbase, directory, 'orthTargets.pt'))
    for i in range(num_classes):
        logging.info('Class {}: {}'.format(i, orthoTarget[i].numpy()))

    # load data
    logging.info('Generating binary codes')
    iters = list(itertools.product(range(2), repeat=args.encode_length))
    codes = []
    for code in iters:
        codes.append(np.array(code) * 2 - 1)

    logging.info('Test code combinations')
    statis = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(num_classes):
        tensor_a = orthoTarget[i].to(device).view(1, -1)
        correct = 0
        correct_dist = 0
        error = 0
        error_dist = 0
        for j in range(int(math.pow(2, args.encode_length))):
            array_b = np.array(codes[j])
            tensor_b = torch.FloatTensor(array_b).to(device).view(1, -1)
            with torch.no_grad():
                server.eval()
                posterior = server([tensor_a, tensor_b])
                predict = posterior.max(1)[1]
                if predict == i:
                    correct += 1
                    correct_dist += ((tensor_a * tensor_b - 1) / -2).sum().item()
                else:
                    error += 1
                    error_dist += ((tensor_a * tensor_b - 1) / -2).sum().item()
            logging.info(
                'Embedding A: {}, Embedding B: {}, Label: {}, Predict: {}'.format(tensor_a.cpu().numpy(),
                                                                                  tensor_b.cpu().numpy(), i,
                                                                                  predict.item()))
        if correct == 0:
            correct = -1
            correct_dist = 1
        elif error == 0:
            error = -1
            error_dist = 1
        logging.info(
            'For Class {}, if predict right, the average hamming distance is {:.2f}; else, the average hamming distance is {:.2f}'.format(
                i, correct_dist / correct, error_dist / error))

else:
    logging.info('Pretrained models do not exist! Please train first.')
