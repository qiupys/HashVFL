import logging

import torch
import math
import numpy as np

import torchvision.models
from torch.nn import BatchNorm1d, Dropout, Linear, Conv2d, Sequential, LSTM, Embedding
from torch.autograd import Function
from transformers import BertModel


def generateNoise(size, epsilon, device):
    if epsilon <= 0:
        return torch.zeros(size).to(device)
    # M(D)=f(D)+Y, Y~Lap(\delta f/epsilon), \delta f is 2.
    return torch.FloatTensor(np.random.laplace(0, 2 / epsilon, size)).to(device)


class GreedyHash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def hash_layer(input):
    return GreedyHash.apply(input)


def myResNet18(in_channels, encode_length):
    model = torchvision.models.resnet18(pretrained=True)
    out_channels = model.conv1.out_channels
    model.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                         bias=False)
    in_features = model.fc.in_features
    model.fc = Linear(in_features, encode_length)
    return model


def myResNet(type, in_channels, encode_length):
    if type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif type == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    else:
        logging.info('Not implemented model type!')
        exit()
    out_channels = model.conv1.out_channels
    model.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                         bias=False)
    in_features = model.fc.in_features
    model.fc = Linear(in_features, encode_length)
    return model


class MyResNet(torch.nn.Module):
    def __init__(self, in_channels, encode_length, defense, epsilon, device):
        super(MyResNet, self).__init__()
        self.base = myResNet18(in_channels, encode_length)
        self.batch_norm = BatchNorm1d(num_features=encode_length)
        self.defense = defense
        self.epsilon = epsilon
        self.device = device

    def forward(self, x):
        x = self.base(x)
        if not self.defense:
            return x, hash_layer(x)
        x = self.batch_norm(x)
        noise = generateNoise(x.size(), self.epsilon, self.device)
        return x, hash_layer(x) + noise


class CustomResNet(torch.nn.Module):
    def __init__(self, type, in_channels, encode_length, defense, device):
        super(CustomResNet, self).__init__()
        self.base = myResNet(type, in_channels, encode_length)
        self.batch_norm = BatchNorm1d(num_features=encode_length)
        self.defense = defense
        self.device = device

    def forward(self, x):
        x = self.base(x)
        if self.defense:
            x = self.batch_norm(x)
        return x, hash_layer(x)


class MyVGGNet(torch.nn.Module):
    def __init__(self, in_channels, encode_length, epsilon, device):
        super(MyVGGNet, self).__init__()
        self.base = torchvision.models.vgg16(pretrained=True)
        self.conv = Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base.features = Sequential(*list(self.base.features.children())[1:24])
        self.fc = Linear(1024, 4096)
        self.base.classifier = Sequential(*list(self.base.classifier.children())[1:])
        self.encode = Linear(1000, encode_length)
        self.batch_norm = BatchNorm1d(num_features=encode_length)
        self.epsilon = epsilon
        self.device = device

    def forward(self, x):
        x = self.conv(x)
        x = self.base.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.base.classifier(x)
        x = self.encode(x)
        x = self.batch_norm(x)
        noise = generateNoise(x.size(), self.epsilon, self.device)
        return x, hash_layer(x) + noise


class BertBaseModel(torch.nn.Module):
    def __init__(self, encode_length, defense, epsilon, device, dropout=0.1):
        super(BertBaseModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = Dropout(dropout)
        self.linear = Linear(768, encode_length)
        self.batch_norm = BatchNorm1d(num_features=encode_length)
        self.defense = defense
        self.epsilon = epsilon
        self.device = device

    def forward(self, tokens, attention_mask=None):
        output = self.bert(tokens, attention_mask=attention_mask)
        dropout_output = self.dropout(output.pooler_output)
        linear_output = self.linear(dropout_output)
        if not self.defense:
            return linear_output, hash_layer(linear_output)
        bn_output = self.batch_norm(linear_output)
        noise = generateNoise(bn_output.size(), self.epsilon, self.device)
        return bn_output, hash_layer(bn_output) + noise


class MyLSTM(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, encode_length, defense, epsilon, device, num_layers=2):
        super(MyLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # vocab_size be consistent with BERT
        self.embedding = Embedding(num_embeddings=30522, embedding_dim=self.embedding_size)
        self.lstm = LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = Linear(in_features=self.hidden_size, out_features=encode_length)
        self.batch_norm = BatchNorm1d(num_features=encode_length)
        self.defense = defense
        self.epsilon = epsilon
        self.device = device

    def forward(self, tokens):
        tokens = tokens.long()
        embeds = self.embedding(tokens)
        output, (hidden_state, cell_state) = self.lstm(embeds)
        hidden_output = hidden_state[-1, :, :].squeeze(0)
        linear_output = self.linear(hidden_output)
        if not self.defense:
            return linear_output, hash_layer(linear_output)
        bn_output = self.batch_norm(linear_output)
        noise = generateNoise(bn_output.size(), self.epsilon, self.device)
        return bn_output, hash_layer(bn_output) + noise


class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features, defense, epsilon, device, num_layers=1):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.defense = defense
        self.epsilon = epsilon
        self.batch_norm = BatchNorm1d(num_features=out_features)
        self.layers = self._make_layers()
        self.device = device

    def forward(self, x):
        x = self.layers(x)
        if not self.defense:
            return x, hash_layer(x)
        x = self.batch_norm(x)
        noise = generateNoise(x.size(), self.epsilon, self.device)
        return x, hash_layer(x) + noise

    def _make_layers(self):
        layers = [torch.nn.Linear(in_features=self.in_features, out_features=int(math.pow(2, 4 + self.num_layers))),
                  torch.nn.ReLU(inplace=True)]
        for i in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(in_features=int(math.pow(2, 4 + self.num_layers - i)),
                                          out_features=int(math.pow(2, 3 + self.num_layers - i))))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(in_features=int(math.pow(2, 5)), out_features=self.out_features))
        return torch.nn.Sequential(*layers)


class Server(torch.nn.Module):
    def __init__(self, num_party, in_features, num_classes, num_layers=1):
        super(Server, self).__init__()
        self.num_party = num_party
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        # self.batch_norm = BatchNorm1d(num_features=num_party * in_features)
        self.layers = self._make_layers()

    def forward(self, embeds):
        # embeds = [LayerNorm(normalized_shape=self.in_features).cuda()(embed) for embed in embeds]
        x = torch.cat(embeds, 1)
        x = self.layers(x)
        return torch.log_softmax(x, dim=1)

    def _make_layers(self):
        layers = [torch.nn.Linear(in_features=self.in_features * self.num_party,
                                  out_features=int(math.pow(2, 3 + self.num_layers))),
                  torch.nn.ReLU(inplace=True)]
        for i in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(in_features=int(math.pow(2, 3 + self.num_layers - i)),
                                          out_features=int(math.pow(2, 2 + self.num_layers - i))))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(in_features=int(math.pow(2, 4)), out_features=self.num_classes))
        return torch.nn.Sequential(*layers)
