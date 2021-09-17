import argparse
import os
import random
import torch
import numpy as np
from dataset import EventData
from trainer import Trainer
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='EventMSE')

parser.add_argument('--numFilters', type=int, default=16, help='num of filters')
parser.add_argument('--batchSize', type=int, default=4, help='batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='num of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='lr, Default=1e-3')
parser.add_argument('--seed', type=int, default=12450, help='seed, Default=12450')
parser.add_argument('--inputDir', type=str, default='./dataset/', help='dataset dir')
parser.add_argument('--saveDir', type=str, default='./model/', help='model save dir')
parser.add_argument('--device', type=str, default='0', help='device')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    # dataloader
    dataset = EventData(args.inputDir)
    train_datasize = int(len(dataset)*0.8)
    test_datasize = len(dataset) - train_datasize
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_datasize, test_datasize])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True)

    # trainer
    trainer = Trainer(train_dataloader, test_dataloader, args.saveDir, nepochs=args.nEpochs, \
        lr=args.lr, seed=args.seed, num_filters=args.numFilters)
    trainer.init_model()
    trainer.train()

if __name__ == '__main__':
    main()