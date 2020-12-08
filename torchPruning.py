import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch_pruning as tp
import argparse
from torch.utils.data import DataLoader
from dataloader import Dataset
import torch.nn as nn
import numpy as np
import pandas

from model import CNNEncoder, RNNDecoder
from loss import *
from activations import *
from dataloader import Dataset
import config

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)

args = parser.parse_args()


def get_dataloader(data_path, testdata_path):
    raw_data = pandas.read_csv(data_path)
    testraw_data = pandas.read_csv(testdata_path)
    train_loader = DataLoader(Dataset(raw_data.to_numpy(), dataDir=os.path.abspath(os.path.dirname(data_path))), **config.dataset_params)
    test_loader = DataLoader(Dataset(testraw_data.to_numpy(), dataDir=os.path.abspath(os.path.dirname(testdata_path))),
                              **config.dataset_params)
    return train_loader, test_loader


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    return correct / total


def train_model(model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = OrdinalCrossEntropy(target, out, device, num_classes=4)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f" % (
                epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f" % (epoch, args.total_epochs, acc))
        if best_acc < acc:
            torch.save(model, 'resnet18-round%d.pth' % (args.round))
            best_acc = acc
        scheduler.step()
    print("Best Acc=%.4f" % (best_acc))


def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 30, 3, 250, 250))

    def prune_conv(conv, pruned_prob):
        weight = conv.weight.detach().cpu().numpy()
        out_channels = weight.shape[0]
        L1_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
        num_pruned = int(out_channels * pruned_prob)
        prune_index = np.argsort(L1_norm)[:num_pruned].tolist()  # remove filters with small L1-Norm
        plan = DG.get_pruning_plan(conv, tp.prune_conv, prune_index)
        plan.exec()

    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    blk_id = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune_fn = tp.prune_conv
        # elif isinstance(m, nn.BatchNorm2d):
        #     prune_fn = tp.prune_batchnorm
        prune_conv(m.conv1, block_prune_probs[blk_id])
        blk_id += 1
    print(model)
    return model


def main():
    data_path = '../five-video-classification-methods/data/data_file_ordinal_logistic_regression_pytorch.csv'
    testdata_path = '../five-video-classification-methods/data/3_combined_test.csv'
    train_loader, test_loader = get_dataloader(data_path, testdata_path)
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    model = nn.Sequential(
        CNNEncoder(**config.cnn_encoder_params),
        RNNDecoder(**config.rnn_decoder_params)
    )
    if args.mode == 'train':
        args.round = 0
        train_model(model, train_loader, test_loader)
    elif args.mode == 'prune':
        previous_ckpt = 'checkpoints/ep-3794-0.193.pth'
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        ckpt = torch.load(previous_ckpt, map_location = map_location)
        model.load_state_dict(ckpt['model_state_dict'])
        prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        train_model(model, train_loader, test_loader)
    elif args.mode == 'test':
        ckpt = 'resnet18-round%d.pth' % (args.round)
        print("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n" % (acc))


if __name__ == '__main__':
    main()