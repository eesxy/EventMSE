import time
import torch
import torch.nn as nn
import numpy as np
from model import UNet_3D
import torch.backends.cudnn as cudnn

# 参考 EVSRCNNTrainer
class Trainer():
    def __init__(self, train_dataloader, test_dataloader, save_dir, nepochs, lr, num_filters, seed=1):
        super(Trainer, self).__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.save_dir = save_dir
        self.lr = lr
        self.seed = seed
        self.nepochs = nepochs
        self.epoch = 0
        self.num_filters = num_filters
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def init_model(self):
        self.model = UNet_3D(num_filters=self.num_filters).to(self.device)
        self.model.weight_init()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.model = nn.DataParallel(self.model)
        self.criterion_1 = torch.nn.L1Loss()
        self.criterion_2 = torch.nn.MSELoss()

        if self.CUDA:
            print('cuda activated')
            # 使用benchmark加速
            cudnn.benchmark = True
            self.criterion_1.cuda()
            self.criterion_2.cuda()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, \
            mode='min', factor=0.3, patience=3, verbose=True, threshold=0.002, \
            threshold_mode='rel', cooldown=2)
                                                            
    def save_model(self, epoch):
        now = time.strftime("%y-%m-%d-%H-%M", time.localtime(time.time()))
        save_path = self.save_dir + f'epoch_{epoch}_' + now + '.pth'
        state = {'model':self.model.module.state_dict(), 'optimizer':self.optimizer.state_dict(), \
            'scheduler':self.scheduler.state_dict(), 'epoch':epoch}
        torch.save(state, save_path)
        print("Checkpoint saved to "+save_path)               

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = np.zeros(len(self.train_dataloader))
        for batch, (data, target) in enumerate(self.train_dataloader):
            data, target = data.type(torch.FloatTensor), target.type(torch.FloatTensor)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data)

            loss_l1 = self.criterion_1(pred, target)
            loss_l2 = self.criterion_2(pred, target)
            loss = 0 * loss_l1 + 1 * loss_l2

            train_loss[batch] = loss.item()
            loss.backward()
            self.optimizer.step()
            if batch % int(len(self.train_dataloader)/10) == 0:
                print(f'epoch {epoch}, {batch} / {len(self.train_dataloader)}, loss: {loss.item()}')
                
        print(f'epoch {epoch}, average loss:{np.average(train_loss)}')
        return train_loss/len(self.train_dataloader)

    def train(self):
        loss_arr = np.zeros(self.nepochs)
        for epoch in range(self.epoch, self.nepochs):
            if self.optimizer.param_groups[0]['lr'] <= 1e-6:
                break
            self.train_epoch(epoch)
            loss_arr[epoch] = self.validate()
            self.save_model(epoch)
            self.scheduler.step(loss_arr[epoch])

        print(f'best epoch: {np.argmin(loss_arr[:epoch])}, {loss_arr[np.argmin(loss_arr[:epoch])]}')

    def validate(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.test_dataloader):
                data, target = data.type(torch.FloatTensor), target.type(torch.FloatTensor)
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)

                loss_l1 = self.criterion_1(pred, target)
                loss_l2 = self.criterion_2(pred, target)
                loss = 0 * loss_l1 + 1 * loss_l2

                test_loss += loss.item()
        print(f'test loss: {test_loss/len(self.test_dataloader)}')
        return test_loss/len(self.test_dataloader)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.model = nn.DataParallel(self.model)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1