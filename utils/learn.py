import logging

import torch
import torch.nn.functional as F

from utils.model import BertBaseModel


def adjust_learning_rate(learning_rate, optimizers, epoch):
    epoch_lr_decrease = 10
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(epoch, workers, server, data_loaders, defense, lr, weight_decay, ortho_target, device):
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

            if defense:
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
                    # cos_loss -= torch.trace(torch.log_softmax(torch.matmul(code, target.t()), dim=1)) / (
                    #         len(code) * len(code[0]))
                cos_loss /= len(codes)

                # # p-norm loss
                # norm_loss = 0
                # for i in range(len(embeds)):
                #     norm_loss += torch.mean(torch.abs(torch.norm(embeds[i] - codes[i], p=3)))
                # norm_loss /= len(embeds)

                loss = cls_loss + cos_loss
            else:
                # classification loss
                preds = server(embeds)
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
    return acc


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
        return acc
