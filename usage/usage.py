import torch
import torch.nn as nn
import numpy as np

# 两次升维
def header_block(in_dim, mid_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(mid_dim),
        nn.LeakyReLU(1e-3),
        nn.Conv3d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 按原文处理, 先保持维度不变, 再升维
def encoder_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(1e-3),
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 2倍上采样
def upsample_block(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 按原文处理, 先降维, 后维数不变卷积
def decoder_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 降维
def tail_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(1e-3),
    )

# 3D UNet, 4次降采样
# num_filter应为2的倍数
class UNet_3D(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, num_filters=4):
        super(UNet_3D, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters

        # header
        self.header = header_block(in_dim, self.num_filters//2, self.num_filters)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # encoder
        self.encoder_1 = encoder_block(self.num_filters, self.num_filters*2)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.encoder_2 = encoder_block(self.num_filters*2, self.num_filters*4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.encoder_3 = encoder_block(self.num_filters*4, self.num_filters*8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # bridge
        self.bridge = encoder_block(self.num_filters*8, self.num_filters*16)

        # decoder
        self.upsample_1 = upsample_block(self.num_filters*16, self.num_filters*16)
        self.decoder_1 = decoder_block(self.num_filters*24, self.num_filters*8)
        self.upsample_2 = upsample_block(self.num_filters*8, self.num_filters*8)
        self.decoder_2 = decoder_block(self.num_filters*12, self.num_filters*4)
        self.upsample_3 = upsample_block(self.num_filters*4, self.num_filters*4)
        self.decoder_3 = decoder_block(self.num_filters*6, self.num_filters*2)
        self.upsample_4 = upsample_block(self.num_filters*2, self.num_filters*2)
        self.decoder_4 = decoder_block(self.num_filters*3, self.num_filters)

        # output
        self.tail = tail_block(self.num_filters, self.out_dim)

    def forward(self, x):
        # x: [n, 1, 32, w*, h*]
        # pad
        w = x.shape[3]
        h = x.shape[4]
        nw = np.ceil(w / 16).astype('int32') * 16
        nh = np.ceil(h / 16).astype('int32') * 16
        inp = nn.functional.pad(x, (0, nh-h, 0, nw-w))
        # x: [n, 1, 32, 16w, 16h]

        # header
        head = self.header(inp) # -> [n, k, 32, 16w, 16h]
        pool_1 = self.pool_1(head) # -> [n, k, 16, 8w, 8h]

        # encoder
        encode_1 = self.encoder_1(pool_1) # -> [n, 2k, 16, 8w, 8h]
        pool_2 = self.pool_2(encode_1) # -> [n, 2k, 8, 4w, 4h]
        encode_2 = self.encoder_2(pool_2) # -> [n, 4k, 8, 4w, 4h]
        pool_3 = self.pool_3(encode_2) # -> [n, 4k, 4, 2w, 2h]
        encode_3 = self.encoder_3(pool_3) # -> [n, 8k, 4, 2w, 2h]
        pool_4 = self.pool_4(encode_3) # -> [n, 8k, 2, w, h]

        # bridge
        bridge = self.bridge(pool_4) # -> [n, 16k, 2, w, h]

        # decoder
        upsample_1 = self.upsample_1(bridge) # -> [n, 16k, 4, 2w, 2h]
        concat_1 = torch.cat([upsample_1, encode_3], dim=1) # -> [n, 24k, 4, 2w, 2h]
        decode_1 = self.decoder_1(concat_1) # -> [n, 8k, 4, 2w, 2h]
        upsample_2 = self.upsample_2(decode_1) # -> [n, 8k, 8, 4w, 4h]
        concat_2 = torch.cat([upsample_2, encode_2], dim=1) # -> [n, 12k, 8, 4w, 4h]
        decode_2 = self.decoder_2(concat_2) # -> [n, 4k, 8, 4w, 4h]
        upsample_3 = self.upsample_3(decode_2) # -> [n, 4k, 16, 8w, 8h]
        concat_3 = torch.cat([upsample_3, encode_1], dim=1) # -> [n, 6k, 16, 8w, 8h]
        decode_3 = self.decoder_3(concat_3) # -> [n, 2k, 16, 8w, 8h]
        upsample_4 = self.upsample_4(decode_3) # -> [n, 2k, 32, 16w, 16h]
        concat_4 = torch.cat([upsample_4, head], dim=1) # -> [n, 3k, 32, 16w, 16h]
        decode_4 = self.decoder_4(concat_4) # -> [n, k, 32, 16w, 16h]

        # output
        out = self.tail(decode_4) # -> [n, 1, 32, 16w, 16h]
        out = out[:, :, :, :x.shape[3], :x.shape[4]]
        return out

    def weight_init(self, a=1e-3):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', a))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class Transformer:
    @classmethod
    def loadtxt(cls, data_path):
        data = np.loadtxt(data_path)
        data[:,0] = data[:,0] - data[0,0]
        data[:,0] = data[:,0] * 1e6
        data = data.astype('int32')
        return data

    @classmethod
    def transform(cls, data, width, height, stacks=32, stack_dense=0.1, threshold=None):
        # shape
        stack_events = np.ceil(width*height*stack_dense).astype('int32')
        stack_num = (np.ceil(data.shape[0] / (stack_events*stacks) * 2) *stacks).astype('int32')

        # indexes
        x = data[:, 1].astype('int32')
        y = data[:, 2].astype('int32')
        z = np.arange(0, stack_num, 2).repeat(stack_events)[:data.shape[0]].astype('int32')
        z = z + 1 - data[:, 3]

        ret = np.zeros((stack_num, height, width), dtype='int32')
        # in-place add
        np.add.at(ret, (z, y, x), 1)
        ret = ret.reshape(-1, stacks, height, width)
        if threshold:
            ret[ret>threshold] = threshold

        return ret

    @classmethod
    def save(cls, data, save_path):
        np.save(save_path, data)


import matplotlib.pyplot as plt
import matplotlib.colors as colors

class Scorer:
    def __init__(self, model_path, stacks=32, model_batch_size=1):
        super(Scorer, self).__init__()
        self.model_path = model_path
        self.stacks = stacks
        self.model_batch_size = model_batch_size
        self.img_batch_size = int(stacks / 2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        check_point = torch.load(model_path)
        self.model = UNet_3D(num_filters=16)
        self.model.load_state_dict(check_point['model'])
        self.model.to(self.device)

    def load(self, data_path):
        data = np.load(data_path)
        return data

    def transform(self, data):
        N = data.shape[0]
        data = np.expand_dims(data, axis=1)
        pred = []
        self.model.eval()
        with torch.no_grad():
            for idx in range(0, N, self.model_batch_size):
                end = idx + self.model_batch_size
                if end > N:
                    end = N
                batch = data[idx:end,...]
                batch = torch.from_numpy(batch).type(torch.FloatTensor)
                batch = batch.to(self.device)
                batch_pred = self.model(batch)
                batch_pred = batch_pred.cpu().numpy()
                pred.append(batch_pred)

        pred = np.concatenate(pred, axis=0)
        pred = np.around(pred)
        pred = pred.squeeze()
        return pred

    def score(self, data, pred):
        dif = data - pred
        score_list = np.sum(dif*dif, axis=(1, 2, 3))
        score_list = score_list / (data.shape[1]*data.shape[2]*data.shape[3])
        return np.average(score_list)

    def plot(self, data, pred):
        pos_stacks = list(range(0, 32, 2))
        neg_stacks = list(range(1, 32, 2))
        input_imgs = data[:, pos_stacks, :, :]-data[:, neg_stacks, :, :]
        pred_imgs = pred[:, pos_stacks, :, :]-pred[:, neg_stacks, :, :]

        labels = np.sum((data-pred)*(data-pred), axis=(2,3))
        labels = labels / (data.shape[2]*data.shape[3])

        norm = colors.Normalize(vmin=-3, vmax=3)
        _, axes = plt.subplots(nrows=self.img_batch_size, ncols=2, figsize=(28, 9*self.img_batch_size))
        for img_idx in range(pred.shape[0]):
            for idx in range(self.img_batch_size):
                input_img = input_imgs[img_idx, idx]
                pred_img = pred_imgs[img_idx, idx]
                label = labels[img_idx, idx]
                
                axes[idx][0].imshow(input_img, norm=norm, cmap='bwr')
                axes[idx][0].axis('off')
                axes[idx][0].set_title('input')
                axes[idx][1].imshow(pred_img, norm=norm, cmap='bwr')
                axes[idx][1].axis('off')
                axes[idx][1].set_title('pred, score: %.4f'%(label))
            plt.savefig(f'plot_{img_idx}.png', bbox_inches='tight')



#########################
#       example         #
#########################
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_path', type=str, help='path of data.txt')
parser.add_argument('model_path', type=str, help='path of model.pth')
parser.add_argument('width', type=int)
parser.add_argument('height', type=int)

args = parser.parse_args()


def main():
    raw = Transformer.loadtxt(args.data_path)
    data = Transformer.transform(raw, args.width, args.height)
    scorer = Scorer(args.model_path)
    pred = scorer.transform(data)
    scorer.plot(data, pred)
    print('score: %.4f'%(scorer.score(data, pred)))
    

if __name__ == '__main__':
    main()